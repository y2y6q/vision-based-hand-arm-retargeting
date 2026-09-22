"""Conservative tooling for a per-application 3DxWare SpaceMouse profile.

The robosuite teleoperation process reads the SpaceMouse through hidapi.  On
Windows, 3DxWare can independently translate the same device reports into
mouse-wheel and menu events for the foreground application.  This module only
works with the user-writable 3DxWare profile directory; it never stops a
service, changes an installed profile, or opens the device.

The profile schema used here is based on the locally installed 3DxWare 10
configuration files for the SpaceMouse Wireless (VID:PID 256F:C62E).  Its
application matching key is an executable basename, so callers must opt in to
each executable that they want to affect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


PROFILE_SUFFIX = "-KMJ.xml"
PROFILE_SCHEMA_VERSION = 1
SPACE_MOUSE_VENDOR_ID = "256f"
SPACE_MOUSE_PRODUCT_ID = "c62e"
SPACE_MOUSE_DEVICE_ID = "ID_ProductID_C62E"
DEFAULT_KMJ_PROFILE = "AppDefCfg_KMJ.xml"
DEFAULT_TARGET_EXECUTABLES = (
    "python.exe",
    "pycharm64.exe",
    "WindowsTerminal.exe",
    "powershell.exe",
    "cmd.exe",
)
AXES = (
    "HIDMultiAxis_X",
    "HIDMultiAxis_Y",
    "HIDMultiAxis_Z",
    "HIDMultiAxis_Rx",
    "HIDMultiAxis_Ry",
    "HIDMultiAxis_Rz",
)
# C62E's two physical buttons are parented to MENU_1 and MENU_2 in the
# installed Base.xml. The direct logical names cover the same driver hierarchy
# on older profiles without altering any raw HID report.
C62E_BUTTON_ACTIONS = (
    "V3DK_MENU_1",
    "V3DK_MENU_2",
    "V3DK_MENU",
    "V3DK_FIT",
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    matches = _children(element, name)
    return matches[0] if matches else None


def _text(element: ET.Element | None, name: str, default: str | None = None) -> str | None:
    value = _child(element, name)
    if value is None or value.text is None:
        return default
    return value.text.strip()


def _find_all(element: ET.Element, name: str) -> list[ET.Element]:
    return [candidate for candidate in element.iter() if _local_name(candidate.tag) == name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_executable_name(executable: str) -> str:
    if not isinstance(executable, str):
        raise ValueError("target executable must be a string")
    value = executable.strip()
    if not value or Path(value).name != value or not value.lower().endswith(".exe"):
        raise ValueError("target executable must be a basename ending in .exe")
    if not re.fullmatch(r"[A-Za-z0-9_. -]+", value):
        raise ValueError("target executable contains unsupported characters")
    return value


def profile_filename(executable: str) -> str:
    """Return 3DxWare's observed user-profile filename for an executable.

    3DxServiceState proposes names such as ``pycharm64-KMJ.xml`` when it
    creates a user profile. Reusing that convention gives the profile the
    installed driver's normal discovery path while its XML ID remains project
    specific.
    """
    executable = _validate_executable_name(executable)
    stem = Path(executable).stem
    return f"{stem}{PROFILE_SUFFIX}"


def _profile_id(executable: str) -> str:
    stem = Path(_validate_executable_name(executable)).stem
    safe = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    return f"ID_HandArmRetargeting_{safe}_KMJ"


@dataclass(frozen=True)
class ThreeDxWarePaths:
    """Relevant 3DxWare paths, injectable for tests and nonstandard installs."""

    install_root: Path | None
    install_cfg: Path | None
    user_cfg: Path
    local_root: Path
    state_file: Path
    programdata_cfg: Path

    @classmethod
    def discover(cls) -> "ThreeDxWarePaths":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        localappdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        programdata = Path(os.environ.get("PROGRAMDATA", r"C:\\ProgramData"))
        candidates = (
            Path(r"E:\\3DxWare\\3DxWinCore"),
            Path(r"C:\\Program Files\\3Dconnexion\\3DxWare\\3DxWinCore"),
            Path(r"C:\\Program Files (x86)\\3Dconnexion\\3DxWare\\3DxWinCore"),
        )
        install_root = next((path for path in candidates if path.is_dir()), None)
        return cls(
            install_root=install_root,
            install_cfg=(install_root / "Cfg") if install_root else None,
            user_cfg=appdata / "3Dconnexion" / "3DxWare" / "Cfg",
            local_root=localappdata / "3Dconnexion" / "3DxWare",
            state_file=localappdata / "3Dconnexion" / "3DxWare" / "3DxServiceState.xml",
            programdata_cfg=programdata / "3Dconnexion" / "3DxWare" / "Cfg",
        )


def _xml_root(path: Path) -> ET.Element | None:
    try:
        return ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None


def _driver_version(paths: ThreeDxWarePaths) -> str | None:
    for candidate in (paths.user_cfg / "Global.xml", paths.install_cfg / "Global.xml" if paths.install_cfg else None):
        if candidate is None or not candidate.is_file():
            continue
        root = _xml_root(candidate)
        if root is None:
            continue
        for element in _find_all(root, "DriverVersion"):
            if element.text and element.text.strip():
                return element.text.strip()
    return None


def _axis_mappings(default_kmj: Path | None) -> list[dict[str, Any]]:
    if default_kmj is None or not default_kmj.is_file():
        return []
    root = _xml_root(default_kmj)
    if root is None:
        return []
    result: list[dict[str, Any]] = []
    for axis in _find_all(root, "Axis"):
        input_node = _child(axis, "Input")
        output_node = _child(axis, "Output")
        input_action = _text(input_node, "ActionID")
        if input_action not in AXES:
            continue
        result.append(
            {
                "input": input_action,
                "output": _text(output_node, "ActionID"),
                "enabled": _text(axis, "Enabled") == "true",
                "scale": _text(output_node, "Scale"),
                "reversed": _text(output_node, "Reversed"),
            }
        )
    return result


def _state_associations(state_file: Path, targets: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    root = _xml_root(state_file)
    if root is None:
        return [], None
    associations: list[dict[str, Any]] = []
    for status in _find_all(root, "AppStatus"):
        app_info = _child(status, "AppInfo")
        executable = _text(app_info, "ExecutableName")
        if not executable or executable.casefold() not in targets:
            continue
        properties = _child(status, "CfgProperties")
        create = _child(app_info, "CreateCfgInfo")
        associations.append(
            {
                "executable": executable,
                "application_name": _text(app_info, "ApplicationName") or _text(app_info, "Name"),
                "last_seen": _text(status, "LastSeen"),
                "profile_id": _text(properties, "ID"),
                "profile_file": _text(properties, "FileName"),
                "proposed_user_profile": _text(create, "Filename"),
            }
        )

    device: dict[str, Any] | None = None
    for entry in _find_all(root, "Device"):
        vendor = (_text(entry, "VendorID") or "").casefold()
        product = (_text(entry, "ProductID") or "").casefold()
        if vendor == SPACE_MOUSE_VENDOR_ID and product == SPACE_MOUSE_PRODUCT_ID:
            device = {
                "name": _text(entry, "Name"),
                "vendor_id": vendor,
                "product_id": product,
                "firmware_version": _text(entry, "FirmwareVersion"),
                "active_state": _text(entry, "ActiveState"),
            }
            break
    return associations, device


def inspect_3dxware(
    paths: ThreeDxWarePaths | None = None,
    target_executables: Sequence[str] = DEFAULT_TARGET_EXECUTABLES,
) -> dict[str, Any]:
    """Read driver state and profile locations without changing 3DxWare."""
    paths = paths or ThreeDxWarePaths.discover()
    targets = {_validate_executable_name(value).casefold() for value in target_executables}
    associations, device = _state_associations(paths.state_file, targets)
    default_kmj = paths.install_cfg / DEFAULT_KMJ_PROFILE if paths.install_cfg else None
    user_profiles = []
    if paths.user_cfg.is_dir():
        user_profiles = sorted(path.name for path in paths.user_cfg.glob("*.xml"))
    installed_targets = {
        executable: (paths.user_cfg / profile_filename(executable)).is_file()
        for executable in target_executables
    }
    return {
        "driver_version": _driver_version(paths),
        "install_root": str(paths.install_root) if paths.install_root else None,
        "install_cfg": str(paths.install_cfg) if paths.install_cfg else None,
        "user_cfg": str(paths.user_cfg),
        "programdata_cfg": str(paths.programdata_cfg),
        "state_file": str(paths.state_file),
        "state_file_present": paths.state_file.is_file(),
        "spacemouse_wireless": device,
        "target_app_associations": associations,
        "user_profiles": user_profiles,
        "tool_profiles_present": installed_targets,
        "default_kmj_axis_mappings": _axis_mappings(default_kmj),
        "known_c62e_button_actions": list(C62E_BUTTON_ACTIONS),
        "profile_scope_note": (
            "3DxWare matches these profiles by executable basename. A python.exe profile "
            "also affects other Python GUI processes; it does not match a project directory."
        ),
    }


def _node(parent: ET.Element, name: str, text: str | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, name, attributes)
    if text is not None:
        child.text = text
    return child


def build_c62e_profile(executable: str) -> str:
    """Build a 3DxWare XML profile that suppresses virtual mouse/menu events.

    The profile disables only 3DxWare's virtual output for the C62E in the
    named foreground executable. It does not claim to modify the physical HID
    interface that ``hidapi`` reads.
    """
    executable = _validate_executable_name(executable)
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root = ET.Element(
        "AppCfg",
        {
            "Default": "false",
            "CfgFormatVersion": "1.3",
            "ThisFileVersion": "1.0",
        },
    )
    root.append(ET.Comment(" Generated by wrist_teleop.threedxware; restore with its manifest. "))
    properties = _node(root, "CfgProperties")
    _node(properties, "ID", _profile_id(executable))
    _node(properties, "Name", f"Hand Arm Retargeting ({executable})")
    _node(properties, "InheritsFromID", "ID_Default_KMJ_Cfg")
    app_info = _node(root, "AppInfo")
    signature = _node(app_info, "Signature")
    _node(signature, "Name", f"Hand Arm Retargeting ({executable})")
    _node(signature, "ExecutableName", executable)
    _node(signature, "Transport", "KMJ")
    settings = _node(root, "Settings")
    _node(settings, "ResponseCurve", "1.0")
    devices = _node(root, "Devices")
    device = _node(devices, "Device")
    _node(device, "ID", SPACE_MOUSE_DEVICE_ID)
    _node(device, "Name", "SpaceMouse Wireless")
    _node(device, "CurrentAxisBank", "HandArmRetargeting")
    axis_bank = _node(device, "AxisBank", Default="true")
    _node(axis_bank, "ID", "HandArmRetargeting")
    _node(axis_bank, "Name", "Hand Arm Retargeting")
    _node(axis_bank, "InheritsFromID", "")
    for action in AXES:
        axis = _node(axis_bank, "Axis")
        _node(axis, "Enabled", "false")
        input_node = _node(axis, "Input")
        _node(input_node, "ActionID", action)
        _node(input_node, "Min", "-512")
        _node(input_node, "Max", "511")
        output_node = _node(axis, "Output")
        _node(output_node, "ActionID", action)
    button_bank = _node(device, "ButtonBank", Default="true")
    _node(button_bank, "ID", "HandArmRetargeting")
    _node(button_bank, "Name", "Hand Arm Retargeting")
    _node(button_bank, "InheritsFromID", "")
    for action in C62E_BUTTON_ACTIONS:
        button = _node(button_bank, "Button")
        input_node = _node(button, "Input")
        _node(input_node, "ActionID", action)
        output_node = _node(button, "Output")
        _node(output_node, "ActionID", "Driver_Disabled")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8") + "\n"


def _backup_manifest(
    paths: ThreeDxWarePaths,
    profile_paths: Sequence[Path],
    backup_root: Path,
) -> tuple[Path, dict[str, Any]]:
    backup_dir = backup_root / f"3dxware-profile-{_timestamp()}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for profile_path in profile_paths:
        if profile_path.is_file():
            backup_file = backup_dir / profile_path.name
            shutil.copy2(profile_path, backup_file)
            entries.append(
                {
                    "target": str(profile_path.resolve()),
                    "original_exists": True,
                    "original_sha256": _sha256(profile_path),
                    "backup_file": backup_file.name,
                }
            )
        else:
            entries.append(
                {
                    "target": str(profile_path.resolve()),
                    "original_exists": False,
                    "original_sha256": None,
                    "backup_file": None,
                }
            )
    manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at_utc": _timestamp(),
        "user_cfg": str(paths.user_cfg.resolve()),
        "entries": entries,
    }
    manifest_path = backup_dir / "manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return manifest_path, manifest


def install_profiles(
    executables: Iterable[str],
    *,
    paths: ThreeDxWarePaths | None = None,
    backup_root: str | Path = "outputs/3dxware_profiles/backups",
    allow_overwrite: bool = False,
) -> Path:
    """Back up then install C62E profiles for explicitly named executables.

    Existing files are never overwritten unless ``allow_overwrite`` is true.
    The returned manifest is required for a guarded restore.
    """
    paths = paths or ThreeDxWarePaths.discover()
    values = tuple(dict.fromkeys(_validate_executable_name(value) for value in executables))
    if not values:
        raise ValueError("at least one --target-exe is required")
    profile_paths = [paths.user_cfg / profile_filename(value) for value in values]
    existing = [path for path in profile_paths if path.exists()]
    if existing and not allow_overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing profile(s): {names}")
    manifest_path, manifest = _backup_manifest(paths, profile_paths, Path(backup_root))
    try:
        for executable, profile_path, entry in zip(values, profile_paths, manifest["entries"]):
            _atomic_write(profile_path, build_c62e_profile(executable).encode("utf-8"))
            entry["installed_sha256"] = _sha256(profile_path)
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    except Exception:
        # The manifest retains enough information for an explicit recovery if a
        # partially completed install needs attention; do not guess at rollback.
        raise
    return manifest_path


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def restore_profiles(
    manifest_path: str | Path,
    *,
    paths: ThreeDxWarePaths | None = None,
    force: bool = False,
) -> list[Path]:
    """Restore profiles from a manifest created by :func:`install_profiles`.

    A restore refuses to replace a profile that changed after installation
    unless callers explicitly pass ``force``. Profiles absent before install
    are removed only when their current digest equals the installed digest.
    """
    paths = paths or ThreeDxWarePaths.discover()
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported 3DxWare backup manifest")
    restored: list[Path] = []
    for entry in manifest.get("entries", []):
        target = Path(entry["target"])
        if (
            not _path_within(target, paths.user_cfg)
            or not re.fullmatch(r"[A-Za-z0-9_. -]+-KMJ\.xml", target.name, flags=re.IGNORECASE)
        ):
            raise ValueError(f"manifest target is outside the managed profile directory: {target}")
        current_matches_install = (
            target.is_file() and entry.get("installed_sha256") == _sha256(target)
        )
        if target.exists() and not current_matches_install and not force:
            raise RuntimeError(
                f"refusing to replace modified profile without --force: {target}"
            )
        if entry.get("original_exists"):
            backup_file = manifest_path.parent / entry["backup_file"]
            if not backup_file.is_file():
                raise FileNotFoundError(f"backup payload missing: {backup_file}")
            _atomic_write(target, backup_file.read_bytes())
        elif target.exists():
            target.unlink()
        restored.append(target)
    return restored
