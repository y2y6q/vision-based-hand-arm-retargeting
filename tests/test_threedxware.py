"""Filesystem-only regression tests for reversible 3DxWare profile tooling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.threedxware import (  # noqa: E402
    AXES,
    C62E_BUTTON_ACTIONS,
    DEFAULT_KMJ_PROFILE,
    ThreeDxWarePaths,
    build_c62e_profile,
    inspect_3dxware,
    install_profiles,
    profile_filename,
    restore_profiles,
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(node, name):
    return [child for child in node if local_name(child.tag) == name]


def child(node, name):
    values = children(node, name)
    return values[0] if values else None


def text(node, name):
    value = child(node, name)
    return value.text if value is not None else None


class ThreeDxWareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.install_cfg = root / "install" / "Cfg"
        self.user_cfg = root / "roaming" / "Cfg"
        self.local_root = root / "local"
        self.install_cfg.mkdir(parents=True)
        self.user_cfg.mkdir(parents=True)
        self.local_root.mkdir(parents=True)
        self.paths = ThreeDxWarePaths(
            install_root=self.install_cfg.parent,
            install_cfg=self.install_cfg,
            user_cfg=self.user_cfg,
            local_root=self.local_root,
            state_file=self.local_root / "3DxServiceState.xml",
            programdata_cfg=root / "programdata" / "Cfg",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_profile_contains_only_disabled_virtual_c62e_axes_and_buttons(self):
        root = ET.fromstring(build_c62e_profile("python.exe"))
        self.assertEqual(local_name(root.tag), "AppCfg")
        device = next(node for node in root.iter() if local_name(node.tag) == "Device")
        self.assertEqual(text(device, "ID"), "ID_ProductID_C62E")
        axis_bank = next(node for node in children(device, "AxisBank"))
        axes = children(axis_bank, "Axis")
        self.assertEqual(len(axes), 6)
        self.assertEqual(
            {text(child(axis, "Input"), "ActionID") for axis in axes}, set(AXES)
        )
        self.assertTrue(all(text(axis, "Enabled") == "false" for axis in axes))
        button_bank = next(node for node in children(device, "ButtonBank"))
        buttons = children(button_bank, "Button")
        self.assertEqual(
            {text(child(button, "Input"), "ActionID") for button in buttons},
            set(C62E_BUTTON_ACTIONS),
        )
        self.assertTrue(all(text(child(button, "Output"), "ActionID") == "Driver_Disabled"
                            for button in buttons))

    def test_profile_filename_is_a_safe_executable_scoped_name(self):
        self.assertEqual(profile_filename("python.exe"), "python-KMJ.xml")
        with self.assertRaises(ValueError):
            profile_filename(r"..\\python.exe")

    def test_install_backups_then_restores_an_existing_profile(self):
        target = self.user_cfg / profile_filename("python.exe")
        original = b"<prior-profile />\n"
        target.write_bytes(original)
        manifest_path = install_profiles(
            ["python.exe"], paths=self.paths,
            backup_root=Path(self.temporary.name) / "backups",
            allow_overwrite=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["entries"][0]["original_sha256"], __import__("hashlib").sha256(original).hexdigest())
        self.assertNotEqual(target.read_bytes(), original)
        restored = restore_profiles(manifest_path, paths=self.paths)
        self.assertEqual(restored, [target])
        self.assertEqual(target.read_bytes(), original)

    def test_restore_removes_only_a_profile_absent_before_install(self):
        target = self.user_cfg / profile_filename("pycharm64.exe")
        manifest_path = install_profiles(
            ["pycharm64.exe"], paths=self.paths,
            backup_root=Path(self.temporary.name) / "backups",
        )
        self.assertTrue(target.exists())
        restore_profiles(manifest_path, paths=self.paths)
        self.assertFalse(target.exists())

    def test_install_refuses_an_existing_profile_without_explicit_overwrite(self):
        target = self.user_cfg / profile_filename("python.exe")
        target.write_text("<profile />", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            install_profiles(
                ["python.exe"], paths=self.paths,
                backup_root=Path(self.temporary.name) / "backups",
            )

    def test_inspection_reads_default_wheel_mapping_and_python_state(self):
        (self.install_cfg / DEFAULT_KMJ_PROFILE).write_text(
            """<AppDefCfg><Devices><Device><AxisBank><Axis><Enabled>true</Enabled>
            <Input><ActionID>HIDMultiAxis_Rx</ActionID></Input>
            <Output><ActionID>HIDMouse_Wheel</ActionID><Scale>1.00</Scale></Output>
            </Axis></AxisBank></Device></Devices></AppDefCfg>""",
            encoding="utf-8",
        )
        self.paths.state_file.write_text(
            """<DriverState><DeviceInfoList><Device><Name>SpaceMouse Wireless</Name>
            <VendorID>256f</VendorID><ProductID>c62e</ProductID></Device></DeviceInfoList>
            <AppStatusList><AppStatus><LastSeen>now</LastSeen><CfgProperties>
            <ID>ID_Default_KMJ_Cfg</ID><FileName>default.xml</FileName></CfgProperties>
            <AppInfo><ExecutableName>python.exe</ExecutableName><ApplicationName>Python</ApplicationName>
            <CreateCfgInfo><Filename>user.xml</Filename></CreateCfgInfo></AppInfo>
            </AppStatus></AppStatusList></DriverState>""",
            encoding="utf-8",
        )
        report = inspect_3dxware(self.paths, ["python.exe"])
        self.assertEqual(report["spacemouse_wireless"]["product_id"], "c62e")
        self.assertEqual(report["target_app_associations"][0]["executable"], "python.exe")
        self.assertEqual(report["default_kmj_axis_mappings"][0]["output"], "HIDMouse_Wheel")
        self.assertTrue(report["tool_profiles_present"]["python.exe"] is False)
