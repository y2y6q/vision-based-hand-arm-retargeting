"""Monocular-camera arm control: the operator's palm drives the end effector.

Pipeline (camera thread -> control loop):

1. ``estimate_hand_pose`` turns one MediaPipe detection into a metric wrist
   pose in the *raw* (unmirrored) camera frame: x right in the raw image,
   y down, z away from the camera. Depth comes from the apparent palm size:
   the six rigid palm landmarks have a known metric 3-D shape (MediaPipe
   world landmarks, averaged over time by ``PalmShapeTemplate``), and the
   palm position is the one that puts every point on its pixel's viewing
   ray (a small linear least-squares problem). This is perspective-exact,
   unaffected by palm tilt or by closing the fingers (finger points are not
   used), and metric at any distance from the camera.
2. ``CameraArmMapper`` filters that pose (One Euro), ignores frames that
   jump implausibly, holds through detection gaps, applies per-axis
   deadbands that only open when that axis is really moving, and turns
   hand motion into an incremental robot target, like a mouse:

   * left / right and up / down: each frame's hand displacement moves the
     target by ``displacement * g(hand speed)``; ``g`` rises from a small
     gain for slow, careful motion to a large one for quick sweeps
     ("pointer acceleration"), so fine positioning and long moves both work.
   * toward / away from the camera (the noisiest axis) is rate controlled:
     hold the hand nearer than the neutral depth to move the robot away,
     farther to bring it back, inside a deadzone it does not move. The
     neutral depth is captured whenever tracking (re)starts.
   * orientation follows the whole palm rotation (knob turn and tilts); it
     accumulates only while the hand is not translating, so the wobble of a
     moving hand never turns the robot.

   Nothing is absolute, so the arm never jumps to a hand position. Showing
   the LEFT hand next to the right one is a clutch (like lifting a mouse):
   the arm holds while the right hand is repositioned.
3. The control loop rate-limits the target (``TargetFollower``) and clamps
   it to a safe workspace / orientation cone (``WorkspaceLimits``);
   ``ArmSafetyGuard`` refuses steps that drive a joint into its limit or the
   arm deeper into a low-manipulability (near-singular) configuration. It
   never moves the arm on its own: a refused step only stops the motion.

Direction convention (operator facing the camera, mirrored selfie view):
moving the hand to your right moves the robot to screen-right of the
simulator's main view, holding it nearer the camera moves the robot away
from the viewer (into the screen), up is up.

Rotations are mapped in the hands' own frames, as if the robot hand were
your hand laid flat: your palm normal <-> the robot palm normal, your
fingers <-> the robot fingers. Turning your hand like a knob in front of the
camera therefore yaws the palm-down robot hand, and tipping your fingers
toward the camera tips the robot fingers down. The knob
motion is measured in the image plane, the most accurate direction for a
single camera.

No rule here acts on the robot by itself: every motion comes from the
operator's hand, and the protections only scale down or stop it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Any, Optional, Sequence

import numpy as np


PALM_LANDMARKS = (0, 1, 5, 9, 13, 17)
# Must be inside the image for a usable pose: wrist and the four knuckles.
# The thumb base (1) is often cut first and is extrapolated well enough.
BORDER_LANDMARKS = (0, 5, 9, 13, 17)

# Raw camera frame (x right in the raw image = operator's left, y down,
# z away from the camera) -> operator control basis [right, away, up].
CAMERA_TO_CONTROL = np.array(((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)))


# ---------------------------------------------------------------------------
# Small rotation helpers (no robosuite dependency).
# ---------------------------------------------------------------------------

def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    k = np.array(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Near 180 degrees: axis from the symmetric part.
        diagonal = np.clip((np.diag(rotation) + 1.0) * 0.5, 0.0, None)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        for other in range(3):
            if other != index:
                axis[other] = math.copysign(axis[other], rotation[index, other] + rotation[other, index])
        return axis / np.linalg.norm(axis) * angle
    axis = np.array((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]))
    return axis / (2.0 * math.sin(angle)) * angle


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    quat = np.asarray(q, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ))


def swing_twist_limit(rotation: np.ndarray, axis: Sequence[float], max_twist: float, max_swing: float) -> tuple[np.ndarray, bool]:
    """Clamp a rotation's twist about ``axis`` and its remaining swing."""

    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    quat = _matrix_to_quat(rotation)
    projection = float(np.dot(quat[1:], axis))
    twist = np.array((quat[0], *(projection * axis)))
    norm = float(np.linalg.norm(twist))
    twist = np.array((1.0, 0.0, 0.0, 0.0)) if norm < 1e-12 else twist / norm
    twist_matrix = _quat_to_matrix(twist)
    swing_matrix = rotation @ twist_matrix.T
    twist_vec = matrix_to_rotvec(twist_matrix)
    swing_vec = matrix_to_rotvec(swing_matrix)
    clamped = False
    twist_angle = float(np.linalg.norm(twist_vec))
    if twist_angle > max_twist:
        twist_vec *= max_twist / twist_angle
        clamped = True
    swing_angle = float(np.linalg.norm(swing_vec))
    if swing_angle > max_swing:
        swing_vec *= max_swing / swing_angle
        clamped = True
    return rotvec_to_matrix(swing_vec) @ rotvec_to_matrix(twist_vec), clamped


# ---------------------------------------------------------------------------
# Palm pose from one MediaPipe detection.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandPose:
    position_m: np.ndarray  # wrist, raw camera frame, metres
    rotation: np.ndarray  # raw camera frame, palm axes as columns
    image_scale: float  # image widths per metre of palm
    timestamp_s: float
    ready: bool = True  # False while the palm-shape template is still warming up
    edge_distance: float = 0.5  # closest wrist / knuckle distance to the image border (frame units)


def _xyz(points: Sequence[Any]) -> np.ndarray:
    values = np.asarray(
        [(p.x, p.y, p.z) if hasattr(p, "x") else p for p in points], dtype=np.float64
    )
    if values.shape[0] != 21 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("expected 21 finite landmarks")
    return values


def palm_frame(world_raw: np.ndarray) -> np.ndarray:
    """Palm axes (columns) from all six palm points of one 3-D frame.

    x: wrist -> mean knuckle, z: palm-plane normal (SVD), y = z x x.
    Averaging over points cuts the noise of MediaPipe's per-landmark depth.
    """

    palm = world_raw[list(PALM_LANDMARKS)]
    centred = palm - palm.mean(axis=0)
    along = world_raw[[5, 9, 13, 17]].mean(axis=0) - world_raw[0]
    across = world_raw[5] - world_raw[17]
    normal = np.linalg.svd(centred, full_matrices=False)[2][2]
    if float(np.dot(normal, np.cross(along, across))) < 0.0:
        normal = -normal
    x_axis = along - np.dot(along, normal) * normal
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        raise ValueError("degenerate palm")
    x_axis /= x_norm
    z_axis = normal / np.linalg.norm(normal)
    return np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))


class PalmShapeTemplate:
    """Time-averaged 3-D palm shape of the current operator, in palm axes.

    A person's palm does not change shape, but each frame's 3-D landmarks
    are noisy (several mm). Using this averaged shape, turned into the
    current palm orientation, makes depth depend on image noise only.
    """

    def __init__(self, alpha: float = 0.01, warmup_samples: int = 30):
        # Running mean for the first ``warmup_samples`` frames, then a very
        # slow update so depth does not creep while the arm is anchored.
        self.alpha = float(alpha)
        self.warmup_samples = int(warmup_samples)
        self.shape: Optional[np.ndarray] = None
        self.samples = 0

    def update(self, world_raw: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        palm = world_raw[list(PALM_LANDMARKS)]
        current = (palm - palm.mean(axis=0)) @ rotation  # palm-local coordinates
        if self.shape is None:
            self.shape = current
        else:
            weight = 1.0 / (self.samples + 1) if self.samples < self.warmup_samples else self.alpha
            self.shape = (1.0 - weight) * self.shape + weight * current
        self.samples += 1
        return self.shape.copy()

    @property
    def ready(self) -> bool:
        return self.samples >= self.warmup_samples

    def reset(self) -> None:
        self.shape = None
        self.samples = 0


def _fit_rotation(local: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Kabsch: rotation taking the template (palm axes) onto this frame's points."""

    u, _, vt = np.linalg.svd(observed.T @ local)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag((1.0, 1.0, d)) @ vt


def _image_angle(points: np.ndarray) -> float:
    along = points[[2, 3, 4, 5]].mean(axis=0) - points[0]  # wrist -> mean knuckle
    return math.atan2(along[1], along[0])


def _project(points: np.ndarray, focal: float, aspect: float) -> np.ndarray:
    return np.column_stack((focal * points[:, 0] / points[:, 2] + 0.5,
                            focal * points[:, 1] / points[:, 2] + 0.5 * aspect))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def estimate_hand_pose(
    image_landmarks: Sequence[Any],
    world_landmarks: Sequence[Any],
    *,
    frame_shape: tuple[int, int],
    mirrored: bool,
    focal_width_normalized: float,
    timestamp_s: float,
    edge_margin: float = 0.01,
    palm_template: Optional[PalmShapeTemplate] = None,
) -> HandPose:
    """Metric wrist pose in the raw camera frame (see module docstring)."""

    image = _xyz(image_landmarks)[:, :2].copy()
    world = _xyz(world_landmarks)[:, :3].copy()
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame shape must be positive")
    border = image[list(BORDER_LANDMARKS)]
    edge_distance = float(np.min(np.minimum(border, 1.0 - border)))
    if edge_distance < edge_margin:
        raise ValueError("palm is cut by the image border")
    aspect = height / width
    if mirrored:
        # The detector saw a horizontally flipped frame: undo it for both the
        # image coordinates and the 3-D landmark chirality.
        image[:, 0] = 1.0 - image[:, 0]
        world[:, 0] = -world[:, 0]
    image[:, 1] *= aspect  # both image axes in units of the frame width
    rotation = palm_frame(world)
    palm_world = world[list(PALM_LANDMARKS)]
    palm_pixels = image[list(PALM_LANDMARKS)]
    rays = np.column_stack((
        (palm_pixels[:, 0] - 0.5) / focal_width_normalized,
        (palm_pixels[:, 1] - 0.5 * aspect) / focal_width_normalized,
        np.ones(len(PALM_LANDMARKS)),
    ))
    system = np.vstack([_skew(ray) for ray in rays])

    def solve_centre(offsets: np.ndarray) -> np.ndarray:
        # Perspective-correct position: palm point i sits at C + d_i (d_i
        # known in camera axes from the metric 3-D shape) and must lie on the
        # viewing ray r_i of its pixel: r_i x (C + d_i) = 0, linear in C.
        # Unlike a weak-perspective scale fit this stays exact when the hand
        # is off-centre and turned (a 4% / 18 mm depth error before).
        rhs = np.concatenate([-_skew(ray) @ offset for ray, offset in zip(rays, offsets)])
        centre, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        if not np.all(np.isfinite(centre)) or centre[2] <= 0.05:
            raise ValueError("implausible palm depth")
        return centre

    if palm_template is not None:
        template = palm_template.update(world, rotation)
        if palm_template.ready:
            rotation = _fit_rotation(template, palm_world - palm_world.mean(axis=0))
        # MediaPipe's per-landmark depth is the noisy part of its 3-D output;
        # the wrist -> knuckles direction in the image is accurate to a pixel.
        # Take the palm's in-image angle (the knob turn mapped to robot yaw)
        # from the pixels: compare with the model's perspective projection and
        # turn the model about the camera axis, faded out as the fingers point
        # toward the camera and that direction stops being measurable.
        measured = _image_angle(palm_pixels)
        weight = float(np.clip((float(np.linalg.norm(rotation[:2, 0])) - 0.3) / 0.4, 0.0, 1.0))
        for _ in range(2):
            centre = solve_centre(template @ rotation.T)
            modelled = _image_angle(_project(centre + template @ rotation.T, focal_width_normalized, aspect))
            correction = (measured - modelled + math.pi) % (2.0 * math.pi) - math.pi
            rotation = rotvec_to_matrix((0.0, 0.0, weight * correction)) @ rotation
        offsets = template @ rotation.T
    else:
        offsets = palm_world - palm_world.mean(axis=0)
    if float(np.linalg.norm(offsets[2] - offsets[5])) < 0.02:  # index MCP - little MCP
        raise ValueError("palm too small")
    centre = solve_centre(offsets)
    # Report the wrist (PALM_LANDMARKS[0]), not the palm centre: people
    # rotate the hand about the wrist, so a pure rotation must not move the arm.
    position = centre + offsets[0]
    scale = focal_width_normalized / float(centre[2])
    return HandPose(position, rotation, scale, float(timestamp_s),
                    ready=palm_template is None or palm_template.ready, edge_distance=edge_distance)


# ---------------------------------------------------------------------------
# Filtering.
# ---------------------------------------------------------------------------

def hand_to_robot_axes(finger_axis: Sequence[float], dorsal_axis: Sequence[float]) -> np.ndarray:
    """Rotation taking operator palm axes (x fingers, y, z back) to robot EE axes."""

    x = np.asarray(finger_axis, dtype=np.float64)
    z = np.asarray(dorsal_axis, dtype=np.float64)
    x /= np.linalg.norm(x)
    z = z - np.dot(z, x) * x
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) for a vector signal.

    Low cutoff at rest removes jitter; the cutoff rises with speed so real
    motion is followed with little lag.
    """

    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float = 1.0):
        self.min_cutoff_hz, self.beta, self.d_cutoff_hz = float(min_cutoff_hz), float(beta), float(d_cutoff_hz)
        self.reset()

    @staticmethod
    def _alpha(cutoff_hz: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def __call__(self, value: Sequence[float], timestamp_s: float) -> np.ndarray:
        x = np.asarray(value, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), float(timestamp_s)
            return x.copy()
        dt = max(float(timestamp_s) - self._t, 1e-3)
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(self.d_cutoff_hz, dt) * (dx - self._dx)
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self._dx))
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        self._t = float(timestamp_s)
        return self._x.copy()


# ---------------------------------------------------------------------------
# Hand -> robot target mapping.
# ---------------------------------------------------------------------------

@dataclass
class CameraArmSettings:
    """Every tuning constant of the camera -> arm mapping, in one place."""

    focal_width_normalized: float = 0.78  # ~65 deg horizontal webcam FOV
    # Left / right and up / down (incremental, pointer acceleration): robot
    # metres per hand metre rise from ``slow_gain`` at ``slow_speed_m_s`` hand
    # speed to ``fast_gain`` at ``fast_speed_m_s``, times ``translation_gain``
    # (the - / = keys).
    translation_gain: float = 1.0
    slow_gain: float = 0.5
    fast_gain: float = 1.8
    slow_speed_m_s: float = 0.02
    fast_speed_m_s: float = 0.12
    # Toward / away from the camera (rate control): robot m/s per metre of
    # hand offset from the neutral depth beyond the deadzone, capped.
    depth_gain: float = 1.5
    depth_deadzone_m: float = 0.03
    depth_max_speed_m_s: float = 0.12
    rotation_gain: float = 1.0  # the whole palm rotation is followed
    # Orientation is frozen while the hand translates faster than this and
    # for ``rotation_freeze_hold_s`` afterwards (a moving hand wobbles).
    rotation_freeze_speed_m_s: float = 0.03
    rotation_freeze_hold_s: float = 0.3
    # Robot end-effector axes (in the OSC site frame) that correspond to the
    # operator's fingers and back of the hand. Shadow (rigid TCP): fingers
    # +z, back of the hand +y (its palmar normal is -y).
    robot_finger_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    robot_dorsal_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    lateral_min_cutoff_hz: float = 1.0
    lateral_beta: float = 15.0
    depth_min_cutoff_hz: float = 0.3
    depth_beta: float = 3.0
    rotation_min_cutoff_hz: float = 0.3
    rotation_beta: float = 0.5
    # Lateral deadband while the hand is still (hand metres).
    lateral_deadband_m: float = 0.004
    # Rotation deadbands in the palm frame: tilts (about fingers / across)
    # depend on MediaPipe depth and are noisier than the in-image knob turn.
    tilt_deadband_rad: float = 0.10
    knob_deadband_rad: float = 0.05
    # An axis counts as moving when the filtered hand travels this far along
    # it within ``motion_window_s``; its deadband then shrinks to 20% so the
    # motion is followed smoothly. Still axes keep their full deadband.
    motion_window_s: float = 0.4
    moving_lateral_m: float = 0.008
    moving_rotation_rad: float = 0.10
    fine_follow_s: float = 0.6
    lost_timeout_s: float = 0.3
    reacquire_frames: int = 15  # ~0.5 s of stable tracking before anchoring
    outlier_jump_m: float = 0.10
    outlier_jump_rad: float = 1.0
    # Left-hand clutch debounce (camera frames with / without the left hand).
    clutch_on_frames: int = 3
    clutch_off_frames: int = 5
    # Wrist / knuckles closer than this to the image border (frame units)
    # are warned about in the camera preview.
    edge_warning: float = 0.08

    def __post_init__(self) -> None:
        for name in ("focal_width_normalized", "translation_gain", "slow_gain", "fast_gain", "depth_gain",
                     "depth_max_speed_m_s", "lateral_min_cutoff_hz", "depth_min_cutoff_hz",
                     "rotation_min_cutoff_hz", "lost_timeout_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.slow_speed_m_s < self.fast_speed_m_s:
            raise ValueError("need 0 <= slow_speed_m_s < fast_speed_m_s")
        if not np.isfinite(self.rotation_gain) or self.rotation_gain < 0.0:
            raise ValueError("rotation_gain must be non-negative")
        if not np.isfinite(self.depth_deadzone_m) or self.depth_deadzone_m < 0.0:
            raise ValueError("depth_deadzone_m must be non-negative")

    def speed_gain(self, hand_speed_m_s: float) -> float:
        """Pointer-acceleration curve: robot metres per hand metre at this speed."""

        x = (float(hand_speed_m_s) - self.slow_speed_m_s) / (self.fast_speed_m_s - self.slow_speed_m_s)
        x = min(max(x, 0.0), 1.0)
        return self.slow_gain + (self.fast_gain - self.slow_gain) * x * x * (3.0 - 2.0 * x)

    def depth_rate(self, offset_m: float) -> float:
        """Robot away-speed (m/s) for a hand held ``offset_m`` nearer the camera."""

        excess = max(abs(float(offset_m)) - self.depth_deadzone_m, 0.0)
        return math.copysign(min(excess * self.depth_gain, self.depth_max_speed_m_s), offset_m) if excess else 0.0


@dataclass
class ArmTargetStatus:
    state: str  # waiting | anchoring | tracking | clutched | lost | paused
    target_pos: Optional[np.ndarray] = None
    target_ori: Optional[np.ndarray] = None
    hand_offset_m: Optional[np.ndarray] = None  # hand travel since anchoring, [right, away, up]
    hand_rotation_rad: Optional[np.ndarray] = None  # palm rotation since anchoring (palm frame)
    details: dict[str, Any] = field(default_factory=dict)


class CameraArmMapper:
    """Thread-safe filtered hand pose -> incremental robot end-effector target."""

    def __init__(self, settings: Optional[CameraArmSettings] = None):
        self.settings = settings or CameraArmSettings()
        s = self.settings
        self._lateral = OneEuroFilter(s.lateral_min_cutoff_hz, s.lateral_beta)
        self._depth = OneEuroFilter(s.depth_min_cutoff_hz, s.depth_beta)
        self._quat = OneEuroFilter(s.rotation_min_cutoff_hz, s.rotation_beta)
        self._lock = threading.Lock()
        self._filtered_pos: Optional[np.ndarray] = None
        self._filtered_rot: Optional[np.ndarray] = None
        self._last_valid_t: Optional[float] = None
        self._good_frames = 0
        self._rejected_frames = 0
        self._paused = False
        self._clutched = False
        self._needs_anchor = True  # wait for stable tracking, then restart from the robot pose
        self._rebase = False  # restart from the robot pose now (clutch release)
        self._hand_ref: Optional[tuple[np.ndarray, np.ndarray]] = None  # hand pose at (re)start
        self._target: Optional[tuple[np.ndarray, np.ndarray]] = None  # accumulated robot target
        self._published: Optional[tuple[np.ndarray, np.ndarray]] = None  # deadbanded hand pose
        self._previous: Optional[tuple[np.ndarray, np.ndarray]] = None  # published pose already used
        self._depth_ref: Optional[float] = None  # neutral depth for rate control
        self._last_tick: Optional[float] = None
        self._history: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._axis_active_until = np.full(2, -1e9)
        self._rot_active_until = -1e9
        self._rot_frozen_until = -1e9
        self._depth_ready = False
        self._last_scale: Optional[float] = None
        self._last_edge: Optional[float] = None

    # -- camera thread -----------------------------------------------------
    def observe(self, pose: Optional[HandPose], timestamp_s: float) -> None:
        with self._lock:
            if pose is None:
                self._good_frames = 0
                return
            quat = _matrix_to_quat(pose.rotation)
            if self._filtered_pos is not None:
                jump = float(np.linalg.norm(pose.position_m - self._filtered_pos))
                previous_quat = _matrix_to_quat(self._filtered_rot)
                turn = 2.0 * math.acos(min(1.0, abs(float(np.dot(quat, previous_quat)))))
                recent = self._last_valid_t is not None and timestamp_s - self._last_valid_t < self.settings.lost_timeout_s
                if recent and (jump > self.settings.outlier_jump_m or turn > self.settings.outlier_jump_rad):
                    self._rejected_frames += 1
                    if self._rejected_frames < 3:
                        return  # single-frame misdetection: ignore it
                    # A persistent jump is a real re-placement of the hand.
                    self._reset_filters_locked()
                    self._needs_anchor = True
                    self._good_frames = 0
                if self._filtered_rot is not None and float(np.dot(quat, _matrix_to_quat(self._filtered_rot))) < 0.0:
                    quat = -quat  # keep the quaternion on one hemisphere for filtering
            self._rejected_frames = 0
            if self._last_valid_t is None or timestamp_s - self._last_valid_t > self.settings.lost_timeout_s:
                # Re-acquired after a gap: start fresh and re-anchor.
                self._reset_filters_locked()
                self._needs_anchor = True
                self._good_frames = 0
            if pose.ready and not self._depth_ready:
                # The palm template just converged: drop depth history built
                # from the less accurate warm-up frames.
                self._depth.reset()
                self._depth_ready = True
            lateral = self._lateral(pose.position_m[:2], timestamp_s)
            depth = self._depth(pose.position_m[2:], timestamp_s)
            filtered_quat = self._quat(quat, timestamp_s)
            self._filtered_pos = np.concatenate((lateral, depth))
            self._filtered_rot = _quat_to_matrix(filtered_quat)
            self._history.append((float(timestamp_s), self._filtered_pos.copy(), self._filtered_rot.copy()))
            horizon = float(timestamp_s) - self.settings.motion_window_s
            while len(self._history) > 2 and self._history[1][0] <= horizon:
                self._history.pop(0)
            self._last_valid_t = float(timestamp_s)
            self._last_scale = pose.image_scale
            self._last_edge = pose.edge_distance
            if pose.ready:
                self._good_frames += 1

    def _reset_filters_locked(self) -> None:
        self._lateral.reset()
        self._depth.reset()
        self._quat.reset()
        self._filtered_pos = None
        self._filtered_rot = None
        self._history = []
        self._axis_active_until = np.full(2, -1e9)
        self._rot_active_until = -1e9
        self._depth_ready = False

    # -- control loop ------------------------------------------------------
    def request_anchor(self) -> None:
        with self._lock:
            self._needs_anchor = True

    def toggle_pause(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            if not self._paused:
                self._needs_anchor = True
            return self._paused

    @property
    def paused(self) -> bool:
        return self._paused

    def set_clutch(self, engaged: bool) -> bool:
        """Clutch in: the arm holds and the hand can be repositioned. Returns True on a change."""

        with self._lock:
            engaged = bool(engaged)
            changed = engaged != self._clutched
            self._clutched = engaged
            return changed

    @property
    def clutched(self) -> bool:
        return self._clutched

    def scale_gains(self, factor: float) -> float:
        with self._lock:
            self.settings.translation_gain = float(np.clip(self.settings.translation_gain * factor, 0.3, 3.0))
            self.settings.depth_gain = float(np.clip(self.settings.depth_gain * factor, 0.3, 4.0))
            return self.settings.translation_gain

    def sync_target(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Replace the accumulated target with what the robot may actually do.

        Called after the workspace clamp, the follow leash and guard blocks,
        so commands beyond a limit are dropped instead of wound up: backing
        away from a limit moves the arm at once.
        """

        with self._lock:
            if self._target is not None:
                self._target = (np.asarray(position, dtype=np.float64).copy(),
                                np.asarray(orientation, dtype=np.float64).copy())

    def reset(self) -> None:
        with self._lock:
            self._reset_filters_locked()
            self._last_valid_t = None
            self._good_frames = 0
            self._needs_anchor = True
            self._rebase = False
            self._hand_ref = self._target = self._published = self._previous = None
            self._depth_ref = self._last_tick = None

    def _publish_locked(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Deadband with hysteresis: ignore small changes at rest, follow real motion."""

        s = self.settings
        published_pos, published_rot = self._published
        old_t, old_pos, old_rot = self._history[0]
        window = self._history[-1][0] - old_t
        travel = self._filtered_pos - old_pos
        lateral_speed = float(np.linalg.norm(travel[:2])) / window if window > 1e-3 else 0.0
        depth_speed = abs(float(travel[2])) / window if window > 1e-3 else 0.0
        axis_moving = np.abs(travel[:2]) > s.moving_lateral_m
        rotating = float(np.linalg.norm(matrix_to_rotvec(self._filtered_rot @ old_rot.T))) > s.moving_rotation_rad
        # Per-axis latch: motion, or crossing the deadband, opens fine
        # following (20% of the band) on that axis for ``fine_follow_s``: a
        # slow move is tracked closely, and noise that crosses once only
        # wobbles briefly before the axis freezes again.
        error = np.abs(self._filtered_pos[:2] - published_pos[:2])
        start = axis_moving | (error > s.lateral_deadband_m)
        self._axis_active_until = np.where(start, timestamp_s + s.fine_follow_s, self._axis_active_until)
        thresholds = np.where(timestamp_s < self._axis_active_until, 0.2 * s.lateral_deadband_m, s.lateral_deadband_m)
        published_pos = published_pos.copy()
        update = np.flatnonzero(error > thresholds)
        published_pos[update] = self._filtered_pos[update]
        # Depth is rate controlled with its own deadzone around the neutral depth.
        published_pos[2] = self._filtered_pos[2]
        rot_band = np.array((s.tilt_deadband_rad, s.tilt_deadband_rad, s.knob_deadband_rad))
        rot_error = matrix_to_rotvec(published_rot.T @ self._filtered_rot)  # palm frame
        rot_change = float(np.linalg.norm(rot_error / rot_band))
        if rotating or rot_change > 1.0:
            self._rot_active_until = timestamp_s + s.fine_follow_s
        if rot_change > (0.2 if timestamp_s < self._rot_active_until else 1.0):
            published_rot = self._filtered_rot.copy()
        self._published = (published_pos, published_rot)
        return published_pos, published_rot, {
            "hand_speed_m_s": lateral_speed, "hand_depth_speed_m_s": depth_speed,
            "hand_moving_axes": axis_moving.tolist(), "hand_rotating": rotating,
        }

    def target(
        self,
        robot_pos: np.ndarray,
        robot_ori: np.ndarray,
        world_from_control: np.ndarray,
        timestamp_s: float,
    ) -> ArmTargetStatus:
        """Robot target for this control tick, or a hold status (no target).

        ``robot_pos`` / ``robot_ori`` is where the arm is commanded now (the
        OSC goal); a (re)start continues from there, so it never jumps.
        """

        s = self.settings
        t = float(timestamp_s)
        with self._lock:
            details: dict[str, Any] = {
                "good_frames": self._good_frames, "image_scale": self._last_scale, "edge_distance": self._last_edge,
                "translation_gain": s.translation_gain, "clutched": self._clutched,
            }
            if self._paused:
                return ArmTargetStatus("paused", details=details)
            fresh = self._last_valid_t is not None and t - self._last_valid_t <= s.lost_timeout_s
            if not fresh or self._filtered_pos is None:
                self._needs_anchor = True
                return ArmTargetStatus("lost" if self._target is not None else "waiting", details=details)
            if self._needs_anchor:
                if self._good_frames < s.reacquire_frames:
                    return ArmTargetStatus("anchoring", details=details)
                self._published = (self._filtered_pos.copy(), self._filtered_rot.copy())
                self._needs_anchor = False
                self._rebase = True
            published_pos, published_rot, motion = self._publish_locked(t)
            details.update(motion)
            if self._clutched:
                # Hold; on release continue from the then-current robot and hand poses.
                self._rebase = True
                return ArmTargetStatus("clutched", details=details)
            if self._rebase:
                self._target = (np.asarray(robot_pos, dtype=np.float64).copy(),
                                np.asarray(robot_ori, dtype=np.float64).copy())
                self._previous = (published_pos.copy(), published_rot.copy())
                self._depth_ref = float(published_pos[2])
                self._hand_ref = (self._filtered_pos.copy(), self._filtered_rot.copy())
                self._last_tick = t
                self._rebase = False
            dt = min(max(t - self._last_tick, 0.0), 0.1)
            self._last_tick = t
            previous_pos, previous_rot = self._previous
            self._previous = (published_pos.copy(), published_rot.copy())

            # Left / right, up / down: this tick's hand displacement times the
            # speed-dependent gain. Depth: velocity from the held offset.
            step = CAMERA_TO_CONTROL @ (published_pos - previous_pos)  # [right, away, up]
            gain = s.speed_gain(motion["hand_speed_m_s"]) * s.translation_gain
            depth_offset = self._depth_ref - float(published_pos[2])  # > 0: hand nearer the camera
            depth_rate = s.depth_rate(depth_offset)
            step = np.array((step[0] * gain, depth_rate * dt, step[2] * gain))

            # Orientation: the whole palm rotation, only while the hand is not translating.
            if (motion["hand_speed_m_s"] > s.rotation_freeze_speed_m_s
                    or motion["hand_depth_speed_m_s"] > 2.0 * s.rotation_freeze_speed_m_s):
                self._rot_frozen_until = t + s.rotation_freeze_hold_s
            frozen = t < self._rot_frozen_until
            turn = matrix_to_rotvec(previous_rot.T @ published_rot)  # palm frame
            turn = np.zeros(3) if frozen else turn * s.rotation_gain

            target_pos, target_ori = self._target
            target_pos = target_pos + np.asarray(world_from_control, dtype=np.float64) @ step
            # The palm-frame turn replayed in the robot hand's own frame.
            correspondence = hand_to_robot_axes(s.robot_finger_axis, s.robot_dorsal_axis)
            target_ori = target_ori @ correspondence @ rotvec_to_matrix(turn) @ correspondence.T
            self._target = (target_pos, target_ori)
            hand_ref_pos, hand_ref_rot = self._hand_ref
            hand_offset = CAMERA_TO_CONTROL @ (self._filtered_pos - hand_ref_pos)
            hand_rotation = matrix_to_rotvec(hand_ref_rot.T @ self._filtered_rot)
            details.update({
                "effective_gain": gain, "depth_offset_m": depth_offset, "depth_rate_m_s": depth_rate,
                "rotation_frozen": frozen,
            })
        return ArmTargetStatus("tracking", target_pos.copy(), target_ori.copy(), hand_offset, hand_rotation, details)


# ---------------------------------------------------------------------------
# Rate limiting, workspace and singularity / joint-limit protection.
# ---------------------------------------------------------------------------

class TargetFollower:
    """Turn an absolute target into bounded per-step world deltas."""

    def __init__(self, max_speed_m_s: float = 0.20, max_rot_speed_rad_s: float = 1.2,
                 settle_m: float = 0.0015, settle_rad: float = 0.01):
        self.max_speed_m_s, self.max_rot_speed_rad_s = float(max_speed_m_s), float(max_rot_speed_rad_s)
        self.settle_m, self.settle_rad = float(settle_m), float(settle_rad)

    def delta(self, target_pos, target_ori, goal_pos, goal_ori, dt: float) -> np.ndarray:
        position_error = np.asarray(target_pos, dtype=np.float64) - np.asarray(goal_pos, dtype=np.float64)
        rotation_error = matrix_to_rotvec(np.asarray(target_ori) @ np.asarray(goal_ori).T)
        distance, angle = float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))
        if distance < self.settle_m:
            position_error = np.zeros(3)
        elif distance > self.max_speed_m_s * dt:
            position_error *= self.max_speed_m_s * dt / distance
        if angle < self.settle_rad:
            rotation_error = np.zeros(3)
        elif angle > self.max_rot_speed_rad_s * dt:
            rotation_error *= self.max_rot_speed_rad_s * dt / angle
        return np.concatenate((position_error, rotation_error))


@dataclass
class WorkspaceLimits:
    """Safe box, reach shell and orientation cone for the end-effector target."""

    box_min: np.ndarray
    box_max: np.ndarray
    shoulder: np.ndarray
    base_axis_xy: np.ndarray
    max_reach_m: float = 0.80
    min_radius_m: float = 0.28
    nominal_ori: np.ndarray = field(default_factory=lambda: np.eye(3))
    max_yaw_rad: float = 2.4
    max_tilt_rad: float = 1.2

    def clamp(self, position: np.ndarray, orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        reasons: list[str] = []
        p = np.clip(position, self.box_min, self.box_max)
        if not np.allclose(p, position):
            reasons.append("workspace_box")
        reach = p - self.shoulder
        distance = float(np.linalg.norm(reach))
        if distance > self.max_reach_m:  # fully stretched arm is singular
            p = self.shoulder + reach * (self.max_reach_m / distance)
            reasons.append("reach_limit")
        radial = p[:2] - self.base_axis_xy
        radius = float(np.linalg.norm(radial))
        if radius < self.min_radius_m:  # wrist above the base axis is singular
            direction = radial / radius if radius > 1e-6 else np.array((1.0, 0.0))
            p = p.copy()
            p[:2] = self.base_axis_xy + direction * self.min_radius_m
            reasons.append("base_axis_limit")
        relative = np.asarray(orientation) @ self.nominal_ori.T
        limited, clamped = swing_twist_limit(relative, (0.0, 0.0, 1.0), self.max_yaw_rad, self.max_tilt_rad)
        if clamped:
            reasons.append("orientation_limit")
        return p, limited @ self.nominal_ori, reasons


class ArmSafetyGuard:
    """Block steps that push a joint into its limit or deepen a singularity.

    Uses the live MuJoCo model: the arm Jacobian at the OSC site, its damped
    pseudo-inverse to predict the joint step, and the manipulability
    ``sqrt(det(J J^T))`` before / after that step. The guard only scales
    down or stops the operator's command; it never moves the arm by itself.
    Moving *away* from a limit or out of a low-manipulability pose is always
    allowed, so the operator can back out instead of the arm locking up.
    """

    def __init__(self, sim: Any, robot: Any, site_name: str, *, joint_margin_rad: float = 0.12,
                 manipulability_fraction: float = 0.25, lookahead_steps: float = 3.0,
                 elbow_index: Optional[int] = 3, elbow_straight_margin_rad: float = 0.75,
                 wall_recheck_s: float = 1.0, clock: Any = None):
        import mujoco

        self._mj = mujoco
        self.model = getattr(sim.model, "_model", sim.model)
        self.live = getattr(sim.data, "_data", sim.data)
        # Simulation time by default: the re-check period is robot time, the
        # same whether or not the computer keeps up with real time.
        self._clock = clock or (lambda: float(self.live.time))
        self.scratch = mujoco.MjData(self.model)
        self.site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if self.site < 0:
            raise ValueError(f"OSC site not found: {site_name}")
        self.qpos_index = np.asarray(robot._ref_joint_pos_indexes, dtype=np.intp)
        self.dof_index = np.asarray(robot._ref_joint_vel_indexes, dtype=np.intp)
        joint_ids = [int(self.model.dof_jntid[i]) for i in self.dof_index]
        self.lower = self.model.jnt_range[joint_ids, 0].copy()
        self.upper = self.model.jnt_range[joint_ids, 1].copy()
        self.joint_margin = float(joint_margin_rad)
        self.lookahead_steps = float(lookahead_steps)
        self.lower_margin = np.full(len(self.dof_index), self.joint_margin)
        self.upper_margin = np.full(len(self.dof_index), self.joint_margin)
        if elbow_index is not None:
            # Straight-elbow singularity: with the 0.0825 m link offsets the Panda
            # arm is stretched (manipulability ~0) at joint 4 ~ -0.47 rad, not at
            # its -0.07 rad limit. Near it the OSC itself pulls the arm in, so the
            # elbow is kept bent at <= -0.82 rad (manipulability ratio > ~0.6),
            # for ~1 cm less reach, instead of being steered out once there.
            self.upper_margin[elbow_index] = float(elbow_straight_margin_rad)
        self.reference_manipulability = self.manipulability(self.live.qpos[self.qpos_index])
        self.min_manipulability = float(manipulability_fraction) * self.reference_manipulability
        # Virtual wall: after a block, the blocked translation direction is
        # removed from later commands, so the arm stops and can slide along
        # it instead of chattering against the limit. It is lifted as soon as
        # the operator commands away from it, after backing off
        # ``wall_release_m``, or when a re-check every ``wall_recheck_s``
        # finds the blocked motion safe again (the pose has changed).
        self.wall_release_m = 0.05
        self.wall_recheck_s = float(wall_recheck_s)
        # Inside a joint margin, at most this much joint motion per step toward
        # the limit is let through (coupling while backing out).
        self.margin_leak_rad = 0.005
        # A blocked step cancels the goal's lead over the arm only when it is
        # larger than this; a smaller one is held instead.
        self.hold_lead_m, self.hold_lead_rad = 0.002, 0.02
        self._wall: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._wall_checked_at = -math.inf

    @property
    def wall_direction(self) -> Optional[np.ndarray]:
        return None if self._wall is None else self._wall[1].copy()

    def _jacobian(self, arm_qpos: np.ndarray) -> np.ndarray:
        self.scratch.qpos[:] = self.live.qpos
        self.scratch.qpos[self.qpos_index] = arm_qpos
        self._mj.mj_kinematics(self.model, self.scratch)
        self._mj.mj_comPos(self.model, self.scratch)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._mj.mj_jacSite(self.model, self.scratch, jacp, jacr, self.site)
        return np.vstack((jacp, jacr))[:, self.dof_index]

    def manipulability(self, arm_qpos: np.ndarray) -> float:
        jacobian = self._jacobian(np.asarray(arm_qpos, dtype=np.float64))
        return float(math.sqrt(max(np.linalg.det(jacobian @ jacobian.T), 0.0)))

    @staticmethod
    def _joint_step(jacobian: np.ndarray, twist: np.ndarray, damping: float = 0.05) -> np.ndarray:
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(6), twist)

    def _evaluate(self, total: np.ndarray, q: np.ndarray, jacobian: np.ndarray, w_now: float,
                  info: dict[str, float], strict: bool = False) -> tuple[float, list[str]]:
        """Largest safe fraction of ``total`` (pose change), and why it was cut.

        ``strict`` also refuses small steps toward a limit from inside its
        margin (normally let through so backing out is never locked).
        """

        if float(np.linalg.norm(total)) < 1e-9:
            return 1.0, []
        qdot = self._joint_step(jacobian, total)
        if w_now < self.min_manipulability:
            # Already near-singular: pass only commands that lead out of it.
            if self.manipulability(q + qdot) > w_now:
                return 1.0, []
            return 0.0, ["singularity"]
        reasons: list[str] = []
        scale = 1.0
        for i, step in enumerate(qdot):
            if step > 0.0:
                room, hard_room = self.upper[i] - self.upper_margin[i] - q[i], self.upper[i] - q[i]
            elif step < 0.0:
                room, hard_room = q[i] - (self.lower[i] + self.lower_margin[i]), q[i] - self.lower[i]
            else:
                continue
            if room > 0.0:
                if abs(step) > room:  # would enter the soft margin: stop at its edge
                    scale = min(scale, room / abs(step))
                    reasons.append(f"joint_limit_{i + 1}")
            elif abs(step) > (1e-6 if strict else min(0.1 * max(hard_room - 0.01, 0.0), self.margin_leak_rad)):
                # Already inside the margin: block only real pressure toward
                # the limit, so a motion that backs out (whose coupling may
                # nudge this joint slightly) is never locked.
                scale = 0.0
                reasons.append(f"joint_limit_{i + 1}")
        if scale > 0.0:
            # Manipulability can collapse within a few steps (e.g. the elbow
            # snapping straight at full reach), so always look ahead three
            # steps instead of only checking once it is already low.
            w_next = self.manipulability(q + self.lookahead_steps * scale * qdot)
            info["manipulability_next"] = w_next
            if w_next < w_now and w_next < self.min_manipulability:
                scale = 0.0
                reasons.append("singularity")
        return scale, reasons

    def check(self, delta: np.ndarray, lead: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[str], dict[str, float]]:
        """Return the allowed delta (possibly scaled or stopped) and reasons.

        ``lead`` is how far the OSC goal already leads the measured pose
        (position, axis-angle). The arm will travel that lead whatever this
        step commands, so the prediction covers ``lead + delta``; when motion
        must stop, the returned delta also cancels the lead, which parks the
        goal on the measured pose instead of letting the arm coast into the
        limit or singularity. Nothing here adds motion of its own.
        """

        delta = np.asarray(delta, dtype=np.float64).copy()
        lead = np.zeros(6) if lead is None else np.asarray(lead, dtype=np.float64)
        position = np.array(self.live.site_xpos[self.site], dtype=np.float64)
        q = self.live.qpos[self.qpos_index].copy()
        jacobian = self._jacobian(q)
        w_now = float(math.sqrt(max(np.linalg.det(jacobian @ jacobian.T), 0.0)))
        info: dict[str, Any] = {
            "manipulability": w_now, "manipulability_ratio": w_now / max(self.reference_manipulability, 1e-12),
            "min_joint_margin_rad": float(np.min(np.minimum(q - self.lower, self.upper - q))),
            "elbow_rad": float(q[3]) if len(q) > 3 else float("nan"),
        }
        wall_reasons: list[str] = []
        if self._wall is not None:
            wall_point, wall_direction = self._wall
            push = float(np.dot(delta[:3], wall_direction))
            backed_off = float(np.dot(position - wall_point, wall_direction)) < -self.wall_release_m
            if push < -1e-6 or backed_off:
                self._wall = None  # the operator moved away from it
            elif push > 0.0:
                now = float(self._clock())
                if now - self._wall_checked_at >= self.wall_recheck_s:
                    self._wall_checked_at = now
                    if self._evaluate(delta + lead, q, jacobian, w_now, {}, strict=True)[0] >= 1.0:
                        self._wall = None  # the blocked motion is clearly safe from here now
                if self._wall is not None:
                    delta[:3] -= push * wall_direction
                    wall_reasons.append("virtual_wall")
            if self._wall is not None:
                info["wall_direction"] = wall_direction.tolist()
        total = delta + lead
        scale, reasons = self._evaluate(total, q, jacobian, w_now, info)
        if not reasons:
            return delta, wall_reasons, info
        translation = float(np.linalg.norm(total[:3]))
        if scale == 0.0 and translation > 1e-6:
            self._wall = (position, total[:3] / translation)
            self._wall_checked_at = float(self._clock())
            info["wall_direction"] = self._wall[1].tolist()
        if scale == 0.0 and float(np.linalg.norm(lead[:3])) < self.hold_lead_m and float(np.linalg.norm(lead[3:])) < self.hold_lead_rad:
            # Stopped with the goal (almost) on the arm: keep the goal where it
            # is so the OSC holds this pose. Re-parking it on the measured pose
            # every step would let the arm drift with its own dynamics.
            return np.zeros(6), wall_reasons + reasons, info
        return scale * total - lead, wall_reasons + reasons, info


# ---------------------------------------------------------------------------
# Camera-worker adapter: one detection drives both the fingers and the arm.
# ---------------------------------------------------------------------------

class CameraArmHandInput:
    """Wrap a finger adapter; also feed the palm pose to ``CameraArmMapper``.

    ``CameraHandWorker`` calls :meth:`update_hand_frame` with the selected
    hand's image and world landmarks from the same MediaPipe result, plus the
    physical labels of every detected hand. Only the expected (physical
    right) hand moves the arm. The other hand appearing next to it is the
    clutch: the arm holds while it is visible (debounced over a few frames).
    """

    def __init__(self, finger_input: Any, mapper: CameraArmMapper, *, camera_mirror: bool,
                 expected_handedness: str = "Right"):
        self.finger_input = finger_input
        self.mapper = mapper
        self.camera_mirror = bool(camera_mirror)
        self.expected_handedness = str(expected_handedness)
        self.last_pose_error: Optional[str] = None
        self.palm_template = PalmShapeTemplate()
        self._clutch_frames = 0  # consecutive frames with (+) / without (-) the clutch hand

    def __getattr__(self, name: str) -> Any:
        # joint_names, calibration_state, smoothing_alpha, ... of the fingers.
        return getattr(self.finger_input, name)

    def _update_clutch(self, detected_hands: Sequence[Any]) -> None:
        labels = {str(label).strip().lower() for label in detected_hands if label}
        expected = self.expected_handedness.lower()
        other = labels - {expected}
        present = expected in labels and bool(other)
        if present:
            self._clutch_frames = max(self._clutch_frames, 0) + 1
        else:
            self._clutch_frames = min(self._clutch_frames, 0) - 1
        settings = self.mapper.settings
        if self._clutch_frames >= settings.clutch_on_frames:
            self.mapper.set_clutch(True)
        elif -self._clutch_frames >= settings.clutch_off_frames:
            self.mapper.set_clutch(False)

    def update_hand_frame(self, image_landmarks, world_landmarks, handedness, timestamp_s, frame_shape,
                          detected_hands: Optional[Sequence[Any]] = None):
        self._update_clutch(detected_hands if detected_hands is not None else (handedness,))
        if world_landmarks is None or str(handedness).strip().lower() != self.expected_handedness.lower():
            self.last_pose_error = "no world landmarks" if world_landmarks is None else f"{handedness} hand ignored"
            self.mapper.observe(None, timestamp_s)
        else:
            try:
                pose = estimate_hand_pose(
                    image_landmarks, world_landmarks, frame_shape=frame_shape, mirrored=self.camera_mirror,
                    focal_width_normalized=self.mapper.settings.focal_width_normalized, timestamp_s=timestamp_s,
                    palm_template=self.palm_template,
                )
                self.mapper.observe(pose, timestamp_s)
                self.last_pose_error = None
            except (TypeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                self.last_pose_error = str(exc)
                self.mapper.observe(None, timestamp_s)
        world_updater = getattr(self.finger_input, "update_world_landmarks", None)
        if callable(world_updater) and world_landmarks is not None:
            return world_updater(world_landmarks, str(handedness), timestamp_s)
        return self.finger_input.update_landmarks(image_landmarks, timestamp_s)

    def mark_lost(self, timestamp_s=None, error=None):
        self._update_clutch(())
        if timestamp_s is not None:
            self.mapper.observe(None, timestamp_s)
        return self.finger_input.mark_lost(timestamp_s, error)

    def latest(self, timestamp_s=None):
        return self.finger_input.latest(timestamp_s)

    def reset(self, timestamp_s=None):
        self.mapper.reset()
        self.mapper.set_clutch(False)
        self._clutch_frames = 0
        self.palm_template.reset()
        return self.finger_input.reset(timestamp_s)
