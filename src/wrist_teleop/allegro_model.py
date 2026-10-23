"""Robosuite 1.5.2 Allegro right-hand gripper backed by the local frozen asset.

This module deliberately reads the existing Panda-gym Allegro right URDF and
mesh files without changing them.  It translates the small, relevant URDF
subset into an in-memory / temporary MJCF document at construction time, then
uses robosuite's normal :class:`GripperModel` composition path.  That keeps
the Panda arm supplied by robosuite intact while preserving the source hand's
joint names, ordering, limits, visual meshes, and collision boxes.

Public integration surface
--------------------------
Import this module before ``suite.make`` and pass
``gripper_types=GRIPPER_TYPE``.  ``make_allegro_panda_env`` is the preferred
factory for this project.  The hand action slice is a 16-vector in ``[-1, 1]``
and maps linearly to the source model's position-actuator joint limits.  Use
``AllegroRightHand.joint_targets_to_action`` for absolute radian targets.

The source model is credited in ``assets/ALLEGRO_PROVENANCE.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URDF = (
    PROJECT_ROOT
    / "third_party"
    / "panda-gym"
    / "panda_gym"
    / "assets"
    / "robots"
    / "panda_allegro"
    / "allegro_hand"
    / "allegro_hand_right.urdf"
)
SOURCE_ASSET_DIR = SOURCE_URDF.parent
GRIPPER_TYPE = "AllegroRightHand"
PALM_DOWN_GRIPPER_TYPE = "AllegroRightHandPalmDown"
_DEFAULT_INITIALIZATION_NOISE = object()

# This is the source URDF declaration order.  Keep it explicit: camera hand
# targets and robosuite action indices both use this exact order.
ALLEGRO_JOINT_NAMES = (
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
)

# The source URDF's main ``base_link`` collision box is centered at this
# location.  It is the physical middle of the broad palm, whereas the source
# ``palm`` link is a virtual wrist-side frame 65 mm behind it.  The broad
# grasp-working face of the physical palm is local +Z.  In Panda's initial
# tabletop pose this is the down-facing side that can see the cube and table;
# the opposite -Z side faces back toward the wrist.  Keep it explicit so the
# visual aids leave the palm working surface instead of the back of the hand.
# The custom MuJoCo hand is intentionally two thirds of the source URDF's
# dimensions.  The source files remain frozen; this value is applied only to
# the generated runtime MJCF, including meshes, collision geometry, body
# offsets, masses, and inertia tensors.
ALLEGRO_GEOMETRY_SCALE = 2.0 / 3.0
_ALLEGRO_SOURCE_PALM_CENTER_LOCAL = (-0.009, 0.0, -0.023)
ALLEGRO_PALM_CENTER_LOCAL = tuple(
    float(value * ALLEGRO_GEOMETRY_SCALE) for value in _ALLEGRO_SOURCE_PALM_CENTER_LOCAL
)
# The physical working face of the generated Allegro palm is local +Z.  This
# shared normal drives both the laser and palm-first-person camera.
ALLEGRO_PALMAR_NORMAL_LOCAL = (0.0, 0.0, 1.0)
# Palm-down teleop mount (same convention as the Shadow entry). In the hand
# root frame the fingers extend along +Z and curl toward +X, so +X is the
# true palm normal. The hand is turned -90 deg about its Y axis so the
# fingers point forward over the table and the palm faces it; the back of the
# palm (root -X face at -20 mm) sits 3 mm off the flange, proximal palm.
ALLEGRO_PALM_DOWN_NORMAL_LOCAL = (1.0, 0.0, 0.0)
_ALLEGRO_PALM_DOWN_TILT_WXYZ = (np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0)
_ALLEGRO_PALM_DOWN_DORSUM_LOCAL = (-0.023, 0.0, -0.045)
# OSC point of the palm-down teleop hand: in front of the palm where curled
# fingertips meet (hand root frame). SpaceMouse rotations then pivot about the
# grasp, instead of swinging the fingers around the flange 10+ cm behind them.
ALLEGRO_PALM_DOWN_GRASP_CENTER_LOCAL = (0.040, 0.0, 0.030)
ALLEGRO_PALM_VIEW_CAMERA = "palm_view_camera"
# This is a palm camera, not a fingertip camera. Keep a small 25 mm clearance
# in front of the palm's working face so MuJoCo does not place the optical
# centre inside palm collision geometry. The optical axis still starts from
# the laser's physical palm centre and points along the same local +Z normal.
ALLEGRO_PALM_VIEW_OFFSET_M = 0.025


@dataclass(frozen=True)
class AllegroJointSpec:
    """Source-provenance record for one controllable Allegro joint."""

    name: str
    lower: float
    upper: float


def _array_text(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rpy_to_wxyz(rpy: Sequence[float]) -> np.ndarray:
    """Convert URDF intrinsic roll/pitch/yaw to MuJoCo's wxyz quaternion."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF roll/pitch/yaw rotation matrix, expressed in the parent body frame."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_attributes(origin: ET.Element | None, *, scale: float = 1.0) -> dict[str, str]:
    """Return MJCF ``pos`` / ``quat`` attributes from an optional URDF origin."""
    if origin is None:
        return {}
    attributes: dict[str, str] = {}
    if origin.get("xyz"):
        xyz = np.fromstring(origin.get("xyz", ""), sep=" ", dtype=np.float64)
        if xyz.shape != (3,):
            raise ValueError(f"Invalid URDF origin xyz: {origin.get('xyz')!r}")
        attributes["pos"] = _array_text(xyz * float(scale))
    if origin.get("rpy"):
        attributes["quat"] = _array_text(_rpy_to_wxyz(np.fromstring(origin.get("rpy", ""), sep=" ")))
    return attributes


def _asset_path(uri: str) -> Path:
    """Resolve the local URDF's mesh URI without accepting arbitrary URIs."""
    cleaned = uri.replace("package://", "")
    path = (SOURCE_ASSET_DIR / cleaned).resolve()
    source_root = SOURCE_ASSET_DIR.resolve()
    if source_root not in path.parents and path != source_root:
        raise ValueError(f"Allegro mesh escapes source asset directory: {uri!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Allegro mesh referenced by URDF is missing: {path}")
    return path


def _joint_specs_from_urdf(root: ET.Element) -> tuple[AllegroJointSpec, ...]:
    specs: list[AllegroJointSpec] = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"Allegro revolute joint {joint.get('name')!r} has no limit")
        specs.append(
            AllegroJointSpec(
                name=str(joint.get("name")),
                lower=float(limit.attrib["lower"]),
                upper=float(limit.attrib["upper"]),
            )
        )
    names = tuple(spec.name for spec in specs)
    if names != ALLEGRO_JOINT_NAMES:
        raise ValueError(f"Unexpected Allegro joint order in {SOURCE_URDF}: {names}")
    return tuple(specs)


def source_joint_specs() -> tuple[AllegroJointSpec, ...]:
    """Read the 16 exact joint names and limits from the frozen source URDF."""
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(
            f"The frozen Allegro source URDF is required but was not found: {SOURCE_URDF}"
        )
    return _joint_specs_from_urdf(ET.parse(SOURCE_URDF).getroot())


def _mesh_name(source: Path, names: dict[Path, str]) -> str:
    if source not in names:
        names[source] = f"mesh_{len(names):02d}"
    return names[source]


def _append_link_content(
    body: ET.Element,
    link: ET.Element,
    mesh_names: dict[Path, str],
    *,
    scale: float,
) -> None:
    """Translate source visual, inertial, and collision data into one MJCF body."""
    inertial = link.find("inertial")
    if inertial is not None:
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is not None and inertia is not None:
            attrs = _origin_attributes(inertial.find("origin"), scale=scale)
            # MuJoCo prohibits an inertial quaternion alongside fullinertia.
            # Rotate the URDF inertia tensor into the body frame instead.
            inertial_origin = inertial.find("origin")
            rpy = np.fromstring(
                inertial_origin.get("rpy", "0 0 0") if inertial_origin is not None else "0 0 0",
                sep=" ",
                dtype=np.float64,
            )
            attrs.pop("quat", None)
            inertia_matrix = np.array(
                [
                    [float(inertia.attrib["ixx"]), float(inertia.attrib.get("ixy", 0.0)), float(inertia.attrib.get("ixz", 0.0))],
                    [float(inertia.attrib.get("ixy", 0.0)), float(inertia.attrib["iyy"]), float(inertia.attrib.get("iyz", 0.0))],
                    [float(inertia.attrib.get("ixz", 0.0)), float(inertia.attrib.get("iyz", 0.0)), float(inertia.attrib["izz"])],
                ],
                dtype=np.float64,
            )
            rotation = _rpy_to_matrix(rpy)
            inertia_matrix = rotation @ inertia_matrix @ rotation.T
            attrs["mass"] = f"{float(mass.attrib['value']) * scale ** 3:.12g}"
            # MuJoCo validates the three principal moments against the rigid
            # body triangle inequality. A few legacy URDF tensors fail that
            # validation. Keep valid tensors exactly; replace only invalid
            # ones with their isotropic mean so the same generated hand can be
            # safely merged into Panda's MJCF (where compiler settings are not
            # carried across by robosuite's merge operation).
            principal = np.linalg.eigvalsh(inertia_matrix)
            if principal[0] <= 0.0 or principal[-1] > float(principal[0] + principal[1]):
                isotropic = max(float(np.mean(np.abs(principal))), 1e-9)
                attrs["diaginertia"] = _array_text([isotropic * scale ** 5] * 3)
            else:
                # MuJoCo fullinertia is Ixx Iyy Izz Ixy Ixz Iyz. The source
                # inertial orientation is represented by the rotation above.
                attrs["fullinertia"] = _array_text(
                    [
                        inertia_matrix[0, 0] * scale ** 5,
                        inertia_matrix[1, 1] * scale ** 5,
                        inertia_matrix[2, 2] * scale ** 5,
                        inertia_matrix[0, 1] * scale ** 5,
                        inertia_matrix[0, 2] * scale ** 5,
                        inertia_matrix[1, 2] * scale ** 5,
                    ]
                )
            ET.SubElement(body, "inertial", attrs)

    for visual_index, visual in enumerate(link.findall("visual")):
        mesh = visual.find("./geometry/mesh")
        if mesh is None:
            continue
        source = _asset_path(mesh.attrib["filename"])
        attrs = _origin_attributes(visual.find("origin"), scale=scale)
        attrs.update(
            type="mesh",
            mesh=_mesh_name(source, mesh_names),
            name=f"{link.get('name')}_visual_{visual_index}",
            group="1",
            contype="0",
            conaffinity="0",
            rgba="0.68 0.68 0.70 1",
        )
        ET.SubElement(body, "geom", attrs)

    for collision_index, collision in enumerate(link.findall("collision")):
        attrs = _origin_attributes(collision.find("origin"), scale=scale)
        attrs.update(
            name=f"{link.get('name')}_collision_{collision_index}",
            group="0",
            contype="1",
            conaffinity="1",
            friction="1 0.005 0.0001",
            condim="4",
        )
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        box = geometry.find("box")
        mesh = geometry.find("mesh")
        if box is not None:
            # URDF boxes use full lengths; MuJoCo's box size is a half extent.
            full_size = np.fromstring(box.attrib["size"], sep=" ", dtype=np.float64)
            if full_size.shape != (3,):
                raise ValueError(f"Invalid Allegro collision box: {box.attrib['size']!r}")
            attrs.update(type="box", size=_array_text(full_size * 0.5 * scale))
        elif mesh is not None:
            source = _asset_path(mesh.attrib["filename"])
            attrs.update(type="mesh", mesh=_mesh_name(source, mesh_names))
        else:
            continue
        ET.SubElement(body, "geom", attrs)


def _quat_mul_wxyz(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def _build_mjcf_tree(palm_down: bool = False) -> ET.ElementTree:
    """Build a self-contained robosuite-compatible MJCF tree from the source URDF.

    ``palm_down`` keeps every link, joint and actuator but places the hand on
    a joint-free ``allegro_mount`` root with the palm facing the table. The
    OSC TCP stays exactly where it was (flange, same orientation), so arm
    control and Gate 2's default model are unaffected.
    """
    source_root = ET.parse(SOURCE_URDF).getroot()
    specs = _joint_specs_from_urdf(source_root)
    source_joints = {joint.get("name"): joint for joint in source_root.findall("joint")}
    links = {link.get("name"): link for link in source_root.findall("link")}
    if "base_link" not in links:
        raise ValueError(f"No base_link in Allegro source: {SOURCE_URDF}")

    children: dict[str, list[ET.Element]] = {}
    for joint in source_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Malformed Allegro joint {joint.get('name')!r}")
        children.setdefault(parent.attrib["link"], []).append(joint)

    model = ET.Element("mujoco", {"model": "allegro_right_hand"})
    ET.SubElement(model, "compiler", {"angle": "radian", "autolimits": "true"})
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "joint", {"damping": "0.01", "armature": "0.001"})
    ET.SubElement(default, "geom", {"solref": "0.02 1", "solimp": "0.9 0.95 0.001"})
    # Position actuators use explicit critical damping.  Without it, the
    # source hand's light distal links overshoot target positions badly under
    # MuJoCo's position servo.
    ET.SubElement(default, "position", {"kp": "20", "dampratio": "1"})
    asset = ET.SubElement(model, "asset")
    actuator = ET.SubElement(model, "actuator")
    sensor = ET.SubElement(model, "sensor")
    contact = ET.SubElement(model, "contact")
    worldbody = ET.SubElement(model, "worldbody")

    # The source integrated Panda-Allegro URDF places base_link 95 mm beyond
    # Panda link 8 with a +135-degree Z mount. At a matched Panda joint pose,
    # installed robosuite's right_hand origin coincides with source link 8;
    # its frame differs by -45 degrees around Z. Therefore the equivalent
    # relative root is +95 mm / +180 degrees about Z. This is retained here
    # rather than silently reusing Panda's two-finger gripper transform.
    root_pos = np.array((0.0, 0.0, 0.095 * ALLEGRO_GEOMETRY_SCALE))
    root_quat = np.array((0.0, 0.0, 0.0, 1.0))
    root_parent = worldbody
    if palm_down:
        mount = ET.SubElement(worldbody, "body", {"name": "allegro_mount", "pos": "0 0 0"})
        root_parent = mount
        root_quat = _quat_mul_wxyz(root_quat, _ALLEGRO_PALM_DOWN_TILT_WXYZ)
        w, x, y, z = root_quat
        rotation = np.array((
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ))
        # The flange face is the mount origin (the default root places the
        # hand's wrist end, root z = -95 mm * scale, exactly there).
        root_pos = -rotation @ np.asarray(_ALLEGRO_PALM_DOWN_DORSUM_LOCAL)
    root_body = ET.SubElement(
        root_parent,
        "body",
        {
            "name": "allegro_right_hand",
            "pos": _array_text(root_pos),
            "quat": _array_text(root_quat),
        },
    )
    palm_normal = ALLEGRO_PALM_DOWN_NORMAL_LOCAL if palm_down else ALLEGRO_PALMAR_NORMAL_LOCAL
    mesh_names: dict[Path, str] = {}

    specs_by_name = {spec.name: spec for spec in specs}

    def add_link(parent_body: ET.Element, link_name: str, source_joint: ET.Element | None) -> ET.Element:
        if link_name not in links:
            raise ValueError(f"Allegro joint references unknown link {link_name!r}")
        attrs = {"name": link_name}
        if source_joint is not None:
            attrs.update(_origin_attributes(source_joint.find("origin"), scale=ALLEGRO_GEOMETRY_SCALE))
        body = ET.SubElement(parent_body, "body", attrs)
        if source_joint is not None and source_joint.get("type") == "revolute":
            name = str(source_joint.get("name"))
            spec = specs_by_name.get(name)
            if spec is None:
                raise ValueError(f"No joint spec for {name!r}")
            axis = source_joint.find("axis")
            joint_attrs = {
                "name": name,
                "type": "hinge",
                "axis": axis.attrib["xyz"] if axis is not None else "0 0 1",
                "limited": "true",
                "range": _array_text([spec.lower, spec.upper]),
                "damping": "0.05",
                "armature": "0.001",
            }
            ET.SubElement(body, "joint", joint_attrs)
        _append_link_content(body, links[link_name], mesh_names, scale=ALLEGRO_GEOMETRY_SCALE)
        for child_joint in children.get(link_name, []):
            child = child_joint.find("child")
            assert child is not None
            add_link(body, child.attrib["link"], child_joint)
        return body

    # Fill the root itself from base_link, then append all URDF child chains.
    _append_link_content(root_body, links["base_link"], mesh_names, scale=ALLEGRO_GEOMETRY_SCALE)
    # This transparent, non-contact marker sits at the geometric centre of
    # the physical palm.  It deliberately does not use the source's virtual
    # ``palm`` / wrist frame, which is offset behind the palm toward the arm.
    ET.SubElement(
        root_body,
        "site",
        {
            "name": "palm_laser_site",
            "pos": _array_text(ALLEGRO_PALM_CENTER_LOCAL),
            "type": "sphere",
            "size": f"{0.002 * ALLEGRO_GEOMETRY_SCALE:.12g}",
            "rgba": "1 0 0 0",
        },
    )
    # MuJoCo cameras look along local -Z.  With x=+X and y=-Y, local -Z is
    # the hand's palm-facing local +Z direction, exactly matching the laser.
    # The camera is immediately in front of the physical palm. MuJoCo cameras
    # look along local -Z; with x=+X and y=-Y this is the palm-facing +Z ray.
    palm_view_position = np.asarray(ALLEGRO_PALM_CENTER_LOCAL) + (
        np.asarray(palm_normal) * ALLEGRO_PALM_VIEW_OFFSET_M
    )
    ET.SubElement(
        root_body,
        "camera",
        {
            "name": ALLEGRO_PALM_VIEW_CAMERA,
            "pos": _array_text(palm_view_position),
            # Palm-down: look along +X with the fingers toward the image top.
            "xyaxes": "0 -1 0 0 0 1" if palm_down else "1 0 0 0 -1 0",
            "fovy": "70",
        },
    )
    for child_joint in children.get("base_link", []):
        child = child_joint.find("child")
        assert child is not None
        add_link(root_body, child.attrib["link"], child_joint)

    # Explicit TCP / EEF: the legacy source's wrist is base_link -95 mm in its
    # local Z direction (palm -65 mm then wrist -30 mm).  OSC references this
    # site after the gripper is merged into Panda's right_hand mount.
    if palm_down:
        # Same orientation as the default model's TCP (root's 180-degree Z
        # mount); position moved to the grasp centre.
        eef_pos = root_pos + rotation @ np.asarray(ALLEGRO_PALM_DOWN_GRASP_CENTER_LOCAL)
        eef = ET.SubElement(mount, "body", {"name": "eef", "pos": _array_text(eef_pos), "quat": "0 0 0 1"})
    else:
        eef = ET.SubElement(root_body, "body", {"name": "eef", "pos": _array_text((0.0, 0.0, -0.095 * ALLEGRO_GEOMETRY_SCALE))})
    ET.SubElement(eef, "site", {"name": "ft_frame", "type": "sphere", "size": f"{0.002 * ALLEGRO_GEOMETRY_SCALE:.12g}", "rgba": "1 0 0 0"})
    ET.SubElement(eef, "site", {"name": "grip_site", "type": "sphere", "size": f"{0.005 * ALLEGRO_GEOMETRY_SCALE:.12g}", "rgba": "1 0 0 0"})
    ET.SubElement(
        eef,
        "site",
        {"name": "grip_site_cylinder", "type": "cylinder", "size": _array_text((0.003 * ALLEGRO_GEOMETRY_SCALE, 0.08 * ALLEGRO_GEOMETRY_SCALE)), "rgba": "0 1 0 0"},
    )
    ET.SubElement(eef, "site", {"name": "ee", "type": "sphere", "size": f"{0.001 * ALLEGRO_GEOMETRY_SCALE:.12g}", "rgba": "0 0 1 0"})
    ET.SubElement(eef, "site", {"name": "ee_x", "pos": _array_text((0.05 * ALLEGRO_GEOMETRY_SCALE, 0.0, 0.0)), "type": "sphere", "size": f"{0.001 * ALLEGRO_GEOMETRY_SCALE:.12g}", "rgba": "1 0 0 0"})
    ET.SubElement(eef, "site", {"name": "ee_y", "pos": _array_text((0.0, 0.05 * ALLEGRO_GEOMETRY_SCALE, 0.0)), "type": "sphere", "size": f"{0.001 * ALLEGRO_GEOMETRY_SCALE:.12g}", "rgba": "0 1 0 0"})
    ET.SubElement(eef, "site", {"name": "ee_z", "pos": _array_text((0.0, 0.0, 0.05 * ALLEGRO_GEOMETRY_SCALE)), "type": "sphere", "size": f"{0.001 * ALLEGRO_GEOMETRY_SCALE:.12g}", "rgba": "0 0 1 0"})

    for source, name in mesh_names.items():
        ET.SubElement(asset, "mesh", {"name": name, "file": str(source), "scale": _array_text((ALLEGRO_GEOMETRY_SCALE,) * 3)})

    # The source URDF intentionally has collision boxes that overlap at every
    # joint interface. MuJoCo reports those as self contacts unless they are
    # explicitly excluded, which makes a hand jump on reset. Exclude only
    # mechanically connected link pairs; all collision geoms still collide
    # with objects, the table, Panda, and non-adjacent hand links.
    for source_joint in source_root.findall("joint"):
        parent = source_joint.find("parent")
        child = source_joint.find("child")
        assert parent is not None and child is not None
        parent_name = parent.attrib["link"]
        if parent_name == "base_link":
            parent_name = "allegro_right_hand"
        ET.SubElement(
            contact,
            "exclude",
            {"body1": parent_name, "body2": child.attrib["link"]},
        )
    for spec in specs:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"allegro_position_{spec.name}",
                "joint": spec.name,
                "ctrllimited": "true",
                "ctrlrange": _array_text([spec.lower, spec.upper]),
                "kp": "20",
                "dampratio": "1",
                "forcelimited": "true",
                "forcerange": "-10 10",
            },
        )
    ET.SubElement(sensor, "force", {"name": "force_ee", "site": "ft_frame"})
    ET.SubElement(sensor, "torque", {"name": "torque_ee", "site": "ft_frame"})
    return ET.ElementTree(model)


def _temporary_mjcf_path(palm_down: bool = False) -> Path:
    """Materialize the generated MJCF only long enough for GripperModel to parse it."""
    tree = _build_mjcf_tree(palm_down)
    handle = tempfile.NamedTemporaryFile(prefix="allegro_right_hand_", suffix=".xml", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        tree.write(path, encoding="unicode", xml_declaration=True)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _target_to_normalized(targets: Sequence[float], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    if values.shape != (len(ALLEGRO_JOINT_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("Allegro joint targets must be 16 finite radians in source joint order")
    clipped = np.clip(values, low, high)
    return np.clip(2.0 * (clipped - low) / (high - low) - 1.0, -1.0, 1.0)


def _normalized_to_target(action: Sequence[float], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (len(ALLEGRO_JOINT_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("Allegro action must be 16 finite normalized values in source joint order")
    normalized = np.clip(values, -1.0, 1.0)
    return low + 0.5 * (normalized + 1.0) * (high - low)


def register_allegro_gripper() -> str:
    """Register the custom model with the installed robosuite gripper factory.

    Importing this module already registers it.  This explicit, idempotent
    helper is useful to callers that want to make the registration visible at
    their environment construction boundary.
    """
    from robosuite.models.grippers import GRIPPER_MAPPING

    GRIPPER_MAPPING[GRIPPER_TYPE] = AllegroRightHand
    GRIPPER_MAPPING[PALM_DOWN_GRIPPER_TYPE] = AllegroRightHandPalmDown
    return GRIPPER_TYPE


from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402


class AllegroRightHand(GripperModel):
    """Sixteen independently position-controlled joints of the right Allegro hand."""

    source_urdf = SOURCE_URDF
    joint_names = ALLEGRO_JOINT_NAMES
    palm_down = False

    def __init__(self, idn: int | str = 0):
        self._joint_specs = source_joint_specs()
        self.lower_limits = np.array([spec.lower for spec in self._joint_specs], dtype=np.float64)
        self.upper_limits = np.array([spec.upper for spec in self._joint_specs], dtype=np.float64)
        mjcf_path = _temporary_mjcf_path(self.palm_down)
        try:
            super().__init__(str(mjcf_path), idn=idn)
        finally:
            mjcf_path.unlink(missing_ok=True)
        if tuple(self._joints) != self.joint_names:
            raise RuntimeError(f"Generated Allegro joint order diverged: {self._joints}")
        if len(self._actuators) != self.dof:
            raise RuntimeError("Generated Allegro MJCF must have one position actuator per joint")

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def init_qpos(self) -> np.ndarray:
        # Source legacy model starts at zero where allowed; thumb joint 12's
        # physical lower limit is positive, so it begins at that valid limit.
        return np.clip(np.zeros(self.dof, dtype=np.float64), self.lower_limits, self.upper_limits)

    @property
    def initial_action(self) -> np.ndarray:
        """Normalized absolute position action that preserves ``init_qpos``."""
        return self.joint_targets_to_action(self.init_qpos)

    def format_action(self, action: Sequence[float]) -> np.ndarray:
        """Validate / clip the direct 16-dimensional normalized position command.

        robosuite's built-in ``GRIP`` controller subsequently maps each value
        linearly into this model's corresponding position actuator ctrlrange.
        It is named ``SimpleGripController`` but its resulting values are sent
        to MuJoCo *position* actuators, not joint-force commands.
        """
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.dof,) or not np.all(np.isfinite(values)):
            return self.initial_action.copy()
        self.current_action = np.clip(values, -1.0, 1.0)
        return self.current_action.copy()

    def joint_targets_to_action(self, targets: Sequence[float]) -> np.ndarray:
        """Map 16 absolute source-order radians into robosuite's ``[-1, 1]`` slice."""
        return _target_to_normalized(targets, self.lower_limits, self.upper_limits)

    def action_to_joint_targets(self, action: Sequence[float]) -> np.ndarray:
        """Map a robosuite normalized hand slice back to actuator target radians."""
        return _normalized_to_target(action, self.lower_limits, self.upper_limits)

    def describe(self) -> dict[str, Any]:
        """Stable metadata for logs and runtime action-slice assertions."""
        return {
            "gripper_type": PALM_DOWN_GRIPPER_TYPE if self.palm_down else GRIPPER_TYPE,
            "mount": "palm_down" if self.palm_down else "fingers_down",
            "geometry_scale": ALLEGRO_GEOMETRY_SCALE,
            "source_urdf": str(self.source_urdf),
            "dof": self.dof,
            "joint_names": list(self.joint_names),
            "lower_limits_rad": self.lower_limits.tolist(),
            "upper_limits_rad": self.upper_limits.tolist(),
            "position_actuators": list(self.actuators),
            "init_qpos_rad": self.init_qpos.tolist(),
            "initial_normalized_action": self.initial_action.tolist(),
            "eef_site": self.important_sites["grip_site"],
        }

    @property
    def _important_geoms(self) -> dict[str, list[str]]:
        # Each entry names actual group=0 geoms generated from the source URDF.
        index = [f"link_{number}.0_collision_0" for number in range(0, 4)]
        middle = [f"link_{number}.0_collision_0" for number in range(4, 8)]
        ring = [f"link_{number}.0_collision_0" for number in range(8, 12)]
        thumb = [f"link_{number}.0_collision_0" for number in range(12, 16)]
        return {
            "left_finger": thumb,
            "right_finger": index + middle + ring,
            "left_fingerpad": ["link_15.0_collision_0"],
            "right_fingerpad": ["link_3.0_collision_0", "link_7.0_collision_0", "link_11.0_collision_0"],
        }

    @property
    def horizontal_radius(self) -> float:
        return 0.13 * ALLEGRO_GEOMETRY_SCALE


# Import-time registration is intentionally explicit: robosuite's factory only
# accepts class names already present in GRIPPER_MAPPING.
class AllegroRightHandPalmDown(AllegroRightHand):
    """Same 16-joint hand and action contract, mounted palm-down for teleop."""

    palm_down = True


register_allegro_gripper()


def make_allegro_panda_env(
    config: Any,
    *,
    has_renderer: bool,
    palm_down: bool = False,
    has_offscreen_renderer: bool = False,
    initialization_noise: Any = _DEFAULT_INITIALIZATION_NOISE,
    hard_reset: bool = True,
    seed: int | None = None,
):
    """Create Panda + the registered 16-DoF Allegro hand with the verified OSC setup.

    ``config`` is the existing wrist ``WristConfig`` object.  This factory is
    intentionally separate from the old wrist-only environment creator so the
    frozen, validated arm-only entry remains unchanged.
    """
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import (
        load_composite_controller_config,
    )

    register_allegro_gripper()
    controller_config = copy.deepcopy(load_composite_controller_config(robot="Panda"))
    arm_config = controller_config["body_parts"]["right"]
    arm_config.update(
        input_type="delta",
        input_ref_frame="world",
        impedance_mode="fixed",
        output_min=[-config.max_translation_delta_m] * 3 + [-config.max_rotation_delta_rad] * 3,
        output_max=[config.max_translation_delta_m] * 3 + [config.max_rotation_delta_rad] * 3,
    )
    # Preserve the live teleoperation factory's historical default when the
    # caller omits this argument.  Scripted validation can explicitly pass
    # ``None`` to remove Panda's default reset noise and make its 3/10 trial
    # protocol reproducible without changing any interactive behavior.
    optional_reset_args = (
        {} if initialization_noise is _DEFAULT_INITIALIZATION_NOISE
        else {"initialization_noise": initialization_noise}
    )
    return suite.make(
        "Lift",
        robots="Panda",
        gripper_types=PALM_DOWN_GRIPPER_TYPE if palm_down else GRIPPER_TYPE,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=False,
        control_freq=int(round(config.control_hz)),
        horizon=10_000_000,
        ignore_done=True,
        hard_reset=bool(hard_reset),
        seed=seed,
        **optional_reset_args,
    )
