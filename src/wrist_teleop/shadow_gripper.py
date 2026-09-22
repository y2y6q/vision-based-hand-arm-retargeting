"""Project-local Panda + Shadow Hand gripper adapter.

This adapter uses the Apache-2.0 Shadow Hand E3M5 right-hand MJCF from
MuJoCo Menagerie.  It is intentionally local to this repository: robosuite's
installed packages remain untouched.  The upstream model has 24 hinge joints
but only 20 position actuators because the distal joints of the four fingers
are tendon-coupled.  Accordingly, robosuite's gripper action slice is exactly
20 normalized values, never an invented 24-value action vector.

The source MJCF uses nested ``<default>`` classes that robosuite 1.5.2 cannot
inline.  ``_robosuite_tree`` resolves those classes before handing the XML to
``GripperModel``.  The generated TCP sites are neutral metadata and do not
change the source hand's joints, collision geometry, mass, friction, or
actuators.
"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets" / "shadow_hand"
SOURCE_MJCF = ASSET_ROOT / "right_hand.xml"
SOURCE_MESH_ROOT = ASSET_ROOT / "assets"
GRIPPER_TYPE = "ShadowHandRight"

# Order in the official right_hand.xml actuator block.  The four *J0* values
# are tendon targets; each moves its corresponding J2/J1 distal pair together.
SHADOW_ACTUATOR_NAMES = (
    "rh_A_WRJ2", "rh_A_WRJ1",
    "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ3", "rh_A_THJ2", "rh_A_THJ1",
    "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
    "rh_A_MFJ4", "rh_A_MFJ3", "rh_A_MFJ0",
    "rh_A_RFJ4", "rh_A_RFJ3", "rh_A_RFJ0",
    "rh_A_LFJ5", "rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0",
)
SHADOW_JOINT_NAMES = (
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
)


def _array_text(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _copy_attrs(element: ET.Element) -> dict[str, str]:
    return {key: value for key, value in element.attrib.items() if key not in {"class", "childclass"}}


def _default_classes(default: ET.Element) -> dict[str, dict[str, dict[str, str]]]:
    """Resolve MuJoCo nested default inheritance into class -> tag -> attrs."""
    classes: dict[str, dict[str, dict[str, str]]] = {}

    def visit(container: ET.Element, inherited: Mapping[str, Mapping[str, str]]) -> None:
        for child in container:
            if child.tag != "default":
                continue
            class_name = child.get("class")
            merged = {tag: dict(attrs) for tag, attrs in inherited.items()}
            for spec in child:
                if spec.tag == "default":
                    continue
                attrs = merged.setdefault(spec.tag, {})
                attrs.update(_copy_attrs(spec))
            if class_name:
                classes[class_name] = merged
            visit(child, merged)

    visit(default, {})
    return classes


def _inline_defaults(root: ET.Element) -> None:
    """Inline explicit / inherited ``class`` and ``childclass`` attributes."""
    default = root.find("default")
    classes = _default_classes(default) if default is not None else {}

    def visit(element: ET.Element, inherited_childclass: str | None = None) -> None:
        selected = element.get("class") or inherited_childclass
        if selected:
            if selected not in classes:
                raise ValueError(f"Unknown Shadow Hand default class {selected!r}")
            for key, value in classes[selected].get(element.tag, {}).items():
                element.attrib.setdefault(key, value)
        element.attrib.pop("class", None)
        childclass = element.attrib.pop("childclass", None) or inherited_childclass
        for child in element:
            visit(child, childclass)

    for top_level in root:
        if top_level.tag != "default":
            visit(top_level)
    if default is not None:
        root.remove(default)


def _robosuite_tree() -> ET.ElementTree:
    """Create an XML tree that robosuite 1.5.2 can prefix and merge."""
    if not SOURCE_MJCF.is_file():
        raise FileNotFoundError(f"Shadow Hand MJCF is missing: {SOURCE_MJCF}")
    if not SOURCE_MESH_ROOT.is_dir():
        raise FileNotFoundError(f"Shadow Hand mesh directory is missing: {SOURCE_MESH_ROOT}")
    tree = ET.parse(SOURCE_MJCF)
    root = tree.getroot()
    _inline_defaults(root)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("./asset/mesh"):
        raw = mesh.get("file")
        if raw is None:
            continue
        source = (SOURCE_MESH_ROOT / raw).resolve()
        if SOURCE_MESH_ROOT.resolve() not in source.parents or not source.is_file():
            raise FileNotFoundError(f"Shadow Hand mesh is missing or outside its asset root: {raw!r}")
        mesh.set("file", str(source))
        # MuJoCo permits an omitted mesh name and derives it from the file
        # stem.  robosuite prefixes references before MuJoCo parses the XML,
        # so make that implicit name explicit first.
        mesh.attrib.setdefault("name", Path(raw).stem)
    # Menagerie reserves group 2 for visual-only meshes and group 3 for
    # collision geometry.  robosuite 1.5.2's XML model helper recognizes only
    # its own visual group 1 and collision group 0, so translate the grouping
    # without changing any geometry shape or collision flags.
    for geom in root.findall(".//geom"):
        if geom.get("group") == "2":
            geom.set("group", "1")
        elif geom.get("group") == "3":
            geom.set("group", "0")

    palm = root.find(".//body[@name='rh_palm']")
    if palm is None:
        raise ValueError("Official Shadow Hand MJCF has no rh_palm body")
    # Keep the source model's grasp reference.  The added child is massless and
    # collision-free; it supplies robosuite's expected EEF / controller sites.
    eef = ET.SubElement(palm, "body", {"name": "eef", "pos": "0 -0.035 0.09"})
    for name, attrs in {
        "ft_frame": {"type": "sphere", "size": "0.002", "rgba": "1 0 0 0"},
        "grip_site": {"type": "sphere", "size": "0.004", "rgba": "1 0 0 0"},
        "grip_site_cylinder": {"type": "cylinder", "size": "0.003 0.08", "rgba": "0 1 0 0"},
        # The fourth entry uses this massless, collision-free marker as the
        # origin of the same world-vertical palm laser shown by Panda +
        # Allegro.  It is deliberately separate from ``grip_site`` so the
        # OSC TCP and the visual targeting aid remain independently named.
        "palm_laser_site": {"type": "sphere", "size": "0.002", "rgba": "1 0 0 0"},
        "ee": {"type": "sphere", "size": "0.001", "rgba": "0 0 1 0"},
        "ee_x": {"pos": "0.05 0 0", "type": "sphere", "size": "0.001", "rgba": "1 0 0 0"},
        "ee_y": {"pos": "0 0.05 0", "type": "sphere", "size": "0.001", "rgba": "0 1 0 0"},
        "ee_z": {"pos": "0 0 0.05", "type": "sphere", "size": "0.001", "rgba": "0 0 1 0"},
    }.items():
        ET.SubElement(eef, "site", {"name": name, **attrs})
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", {"name": "force_ee", "site": "ft_frame"})
    ET.SubElement(sensor, "torque", {"name": "torque_ee", "site": "ft_frame"})
    return tree


def _temporary_mjcf_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="shadow_hand_right_", suffix=".xml", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        _robosuite_tree().write(path, encoding="unicode", xml_declaration=True)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _actuator_ranges_from_tree(tree: ET.ElementTree) -> tuple[np.ndarray, np.ndarray]:
    ranges: list[tuple[float, float]] = []
    names: list[str] = []
    for actuator in tree.getroot().findall("./actuator/*"):
        name = actuator.get("name")
        ctrlrange = actuator.get("ctrlrange")
        if name is None or ctrlrange is None:
            raise ValueError("Every Shadow position actuator must have a name and ctrlrange")
        values = np.fromstring(ctrlrange, sep=" ", dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or not values[0] < values[1]:
            raise ValueError(f"Invalid Shadow actuator ctrlrange {name!r}: {ctrlrange!r}")
        names.append(name)
        ranges.append((float(values[0]), float(values[1])))
    if tuple(names) != SHADOW_ACTUATOR_NAMES:
        raise ValueError(f"Shadow actuator order changed: {tuple(names)!r}")
    return np.asarray(ranges, dtype=np.float64)[:, 0], np.asarray(ranges, dtype=np.float64)[:, 1]


def _joint_initial_qpos_from_tree(tree: ET.ElementTree) -> np.ndarray:
    values: list[float] = []
    names: list[str] = []
    for joint in tree.getroot().findall(".//joint"):
        name = joint.get("name")
        if name is None:
            continue
        raw = joint.get("range")
        if raw is None:
            raise ValueError(f"Shadow joint {name!r} has no range")
        limits = np.fromstring(raw, sep=" ", dtype=np.float64)
        if limits.shape != (2,) or not limits[0] < limits[1]:
            raise ValueError(f"Invalid Shadow joint range {name!r}: {raw!r}")
        names.append(name)
        values.append(float(np.clip(0.0, limits[0], limits[1])))
    if tuple(names) != SHADOW_JOINT_NAMES:
        raise ValueError(f"Shadow joint order changed: {tuple(names)!r}")
    return np.asarray(values, dtype=np.float64)


def register_shadow_gripper() -> str:
    """Register the local adapter with the live robosuite factory only."""
    from robosuite.models.grippers import GRIPPER_MAPPING

    GRIPPER_MAPPING[GRIPPER_TYPE] = ShadowHandRight
    return GRIPPER_TYPE


from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402


class ShadowHandRight(GripperModel):
    """Official right Shadow Hand: 24 joint states, 20 normalized actuators."""

    source_mjcf = SOURCE_MJCF
    joint_names = SHADOW_JOINT_NAMES
    actuator_names = SHADOW_ACTUATOR_NAMES

    def __init__(self, idn: int | str = 0):
        tree = _robosuite_tree()
        self.lower_limits, self.upper_limits = _actuator_ranges_from_tree(tree)
        self._init_qpos = _joint_initial_qpos_from_tree(tree)
        path = _temporary_mjcf_path()
        try:
            super().__init__(str(path), idn=idn)
        finally:
            path.unlink(missing_ok=True)
        if tuple(self._joints) != self.joint_names:
            raise RuntimeError(f"Shadow Hand joint order diverged: {self._joints!r}")
        if tuple(self._actuators) != self.actuator_names:
            raise RuntimeError(f"Shadow Hand actuator order diverged: {self._actuators!r}")

    @property
    def dof(self) -> int:
        return len(self.actuator_names)

    @property
    def init_qpos(self) -> np.ndarray:
        return self._init_qpos.copy()

    @property
    def initial_action(self) -> np.ndarray:
        return self.targets_to_action(np.clip(np.zeros(self.dof), self.lower_limits, self.upper_limits))

    def format_action(self, action: Sequence[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.dof,) or not np.all(np.isfinite(values)):
            return self.initial_action.copy()
        self.current_action = np.clip(values, -1.0, 1.0)
        return self.current_action.copy()

    def targets_to_action(self, targets: Sequence[float]) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (self.dof,) or not np.all(np.isfinite(values)):
            raise ValueError("Shadow Hand needs 20 finite actuator targets in official actuator order")
        clipped = np.clip(values, self.lower_limits, self.upper_limits)
        return np.clip(2.0 * (clipped - self.lower_limits) / (self.upper_limits - self.lower_limits) - 1.0, -1.0, 1.0)

    def action_to_targets(self, action: Sequence[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.dof,) or not np.all(np.isfinite(values)):
            raise ValueError("Shadow Hand needs 20 finite normalized actions")
        values = np.clip(values, -1.0, 1.0)
        return self.lower_limits + 0.5 * (values + 1.0) * (self.upper_limits - self.lower_limits)

    @property
    def _important_geoms(self) -> dict[str, list[str]]:
        # Official source is a five-finger hand.  Keep the thumb separate so
        # robosuite's generic grasp helpers retain their opposition grouping.
        return {
            "left_finger": ["rh_thdistal_g0_col"],
            "right_finger": ["rh_ffdistal_g0_col", "rh_mfdistal_g0_col", "rh_rfdistal_g0_col", "rh_lfdistal_g0_col"],
            "left_fingerpad": ["rh_thdistal_g0_col"],
            "right_fingerpad": ["rh_ffdistal_g0_col", "rh_mfdistal_g0_col", "rh_rfdistal_g0_col", "rh_lfdistal_g0_col"],
        }

    @property
    def horizontal_radius(self) -> float:
        return 0.18

    def describe(self) -> dict[str, Any]:
        return {
            "gripper_type": GRIPPER_TYPE,
            "source_mjcf": str(self.source_mjcf),
            "physical_joint_count": len(self.joint_names),
            "action_dof": self.dof,
            "joint_names": list(self.joint_names),
            "actuator_names": list(self.actuator_names),
            "tendon_coupled_actuators": ["rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"],
            "lower_limits": self.lower_limits.tolist(),
            "upper_limits": self.upper_limits.tolist(),
            "eef_site": self.important_sites["grip_site"],
        }


def make_shadow_panda_env(config: Any, *, has_renderer: bool, has_offscreen_renderer: bool = False):
    """Construct the smallest Panda OSC_POSE + Shadow Hand Lift environment."""
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

    register_shadow_gripper()
    controller_config = copy.deepcopy(load_composite_controller_config(robot="Panda"))
    arm = controller_config["body_parts"]["right"]
    arm.update(
        input_type="delta",
        input_ref_frame="world",
        impedance_mode="fixed",
        output_min=[-config.max_translation_delta_m] * 3 + [-config.max_rotation_delta_rad] * 3,
        output_max=[config.max_translation_delta_m] * 3 + [config.max_rotation_delta_rad] * 3,
    )
    return suite.make(
        "Lift",
        robots="Panda",
        gripper_types=GRIPPER_TYPE,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=False,
        control_freq=int(round(config.control_hz)),
        horizon=10_000,
        ignore_done=True,
    )


# The fourth entry imports this project-local module before it asks robosuite
# to resolve ``gripper_types=\"ShadowHandRight\"``.  Register at that import
# boundary so direct ``suite.make`` callers and hard resets see the same class.
# No installed robosuite file is changed.
register_shadow_gripper()
