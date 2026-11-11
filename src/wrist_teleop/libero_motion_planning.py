"""Approach planning for the LIBERO planner teleop entry (approach, then the operator grasps).

The operator picks a numbered task object or fixture. The planner moves the
Panda end effector until the hand's grasp space is above that target. While it
plans or moves, any SpaceMouse input cancels it and the operator has the arm
again. Only one of them commands the arm at a time; nothing is blended. The
camera hand drives the Shadow fingers the whole time.

Geometry used here (all world frame, metres):

- fingertip: the end of each distal phalanx's collision mesh (its bounding-box
  face farthest from the distal joint);
- grasp centre: midway between the thumb tip and the mean of the index and
  middle tips. Thumb and index close in a two-finger pinch, a tripod and a
  five-finger grasp alike, so the point sits in the grasp space for all of
  them and moves with the pre-shape the operator holds when choosing a target;
- finger roots: where the index, middle and ring fingers leave the palm (their
  MCP flexion joints, the proximal phalanx origins);
- aim point: with the palm facing down (the forward mount, palm parallel to
  the table) the centre of those three finger roots, so the object ends up
  under the base of the fingers. The grasp centre left it under the
  fingertips, and the operator pushed the hand ~5 cm further along the fingers
  after every arrival (2026-11-10). Otherwise (the down mount: fingers down,
  the roots straight above the tips) the grasp centre;
- target box: the axis-aligned box around the target's collision geoms;
- approach goal: the aim point (or a held object's centre) above the
  target's centre, with the lowest fingertip (or held object's bottom)
  ``clearance_m`` above the target's top. The wrist orientation is kept.

The ``ompl`` backend runs in a separate Python (``scripts/ompl_planner_worker.py``
in the ``ompl_planner`` Conda env) on the same compiled scene. It first
checks the straight-line routes from ``cartesian_candidates`` (lift /
traverse / descend, then one diagonal to above the goal and descend; hand
orientation fixed, IK continued every centimetre so the arm cannot switch
posture, MuJoCo clearance to the whole robot) and only if all are blocked
searches with OMPL RRT-Connect, the hand orientation kept within a tube.
``straight`` drives the first lift / traverse / descend route without any
check, for comparison.

Pure-numpy parts (geometry, paths) import without MuJoCo or LIBERO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OMPL_WORKER = ROOT / "scripts" / "ompl_planner_worker.py"
OMPL_ENV_NAME = "ompl_planner"

FINGERS = ("th", "ff", "mf", "rf", "lf")
# With the palm facing down, the object goes under the centre of these finger
# roots; "facing down" = the palm normal within this angle of straight down.
ROOT_FINGERS = ("ff", "mf", "rf")
PALM_DOWN_MAX_TILT_DEG = 45.0
# Lowest fingertip (or held object's bottom) above the target's top at arrival.
PREGRASP_CLEARANCE_M = 0.02
# Extra height over anything the straight path passes above.
STRAIGHT_LIFT_MARGIN_M = 0.05
# The checked planner tries these lift margins in turn (and a direct line first
# when the goal is this close, e.g. pressing the same number again to correct).
CARTESIAN_LIFT_MARGINS_M = (0.05, 0.10, 0.16)
DIRECT_PATH_WITHIN_M = 0.05
# Peak planner speeds, in simulated time (slow motion slows them in real time).
# 2026-11-06: 0.15 -> 0.30 m/s after the first real run ("too slow"); the
# angular peak keeps the old ratio. ``--planner-speed`` sets the linear one.
PLANNER_MAX_SPEED_M_S = 0.30
PLANNER_SPEED_RANGE_M_S = (0.05, 0.5)
PLANNER_ANGULAR_PER_LINEAR = 0.8 / 0.15
PLANNER_MAX_ANGULAR_SPEED_RAD_S = PLANNER_MAX_SPEED_M_S * PLANNER_ANGULAR_PER_LINEAR
# Arrival: the end effector is this close to the final goal (or the time runs out).
ARRIVAL_TOLERANCE_M = 0.008
ARRIVAL_TOLERANCE_RAD = 0.05
ARRIVAL_TIMEOUT_S = 1.5
PLANNER_BACKENDS = ("ompl", "straight")
# Peak / mean speed of a minimum-jerk profile.
_MIN_JERK_PEAK_RATIO = 1.875


# --------------------------------------------------------------- rotations
def mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""

    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + fraction * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1.0 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / np.sin(theta)


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the rotation taking ``a`` to ``b``."""

    relative = np.asarray(b, dtype=np.float64).reshape(3, 3) @ np.asarray(a, dtype=np.float64).reshape(3, 3).T
    return float(np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0)))


# ---------------------------------------------------------------- geometry
@dataclass(frozen=True)
class TargetBox:
    """Axis-aligned world box around one target's collision geoms."""

    lower: np.ndarray
    upper: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def top(self) -> float:
        return float(self.upper[2])


def grasp_center(tips: dict[str, np.ndarray]) -> np.ndarray:
    """Midway between the thumb tip and the mean of the index and middle tips."""

    return 0.5 * (np.asarray(tips["th"], dtype=np.float64)
                  + 0.5 * (np.asarray(tips["ff"], dtype=np.float64) + np.asarray(tips["mf"], dtype=np.float64)))


def finger_root_center(roots: dict[str, np.ndarray]) -> np.ndarray:
    """Centre of the index, middle and ring finger roots."""

    return np.mean([np.asarray(roots[finger], dtype=np.float64) for finger in ROOT_FINGERS], axis=0)


def palm_faces_down(palm_normal: Optional[np.ndarray], max_tilt_deg: float = PALM_DOWN_MAX_TILT_DEG) -> bool:
    """True when the palm normal is within ``max_tilt_deg`` of straight down."""

    if palm_normal is None:
        return False
    normal = np.asarray(palm_normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    return length > 0.0 and -float(normal[2]) / length >= float(np.cos(np.radians(max_tilt_deg)))


@dataclass(frozen=True)
class HandState:
    """What the approach goal needs from the current hand (and any held object)."""

    ee_pos: np.ndarray
    ee_rot: np.ndarray
    tips: dict[str, np.ndarray]
    held: Optional[TargetBox] = None  # union box of held objects
    roots: Optional[dict[str, np.ndarray]] = None  # finger roots (ROOT_FINGERS), world
    palm_normal: Optional[np.ndarray] = None  # world direction the palm faces

    @property
    def aims_with_roots(self) -> bool:
        return self.roots is not None and palm_faces_down(self.palm_normal)

    @property
    def aim_point(self) -> np.ndarray:
        """The hand point put over an object: the finger-root centre with the palm down, else the grasp centre."""
        return finger_root_center(self.roots) if self.aims_with_roots else grasp_center(self.tips)

    @property
    def aim(self) -> str:
        """Which point is aligned over the target (for the log)."""
        if self.held is not None:
            return "held object"
        return "finger roots" if self.aims_with_roots else "grasp centre"

    @property
    def reference_point(self) -> np.ndarray:
        """The point aligned over the target: the held object's centre, else the aim point."""
        return self.held.center if self.held is not None else self.aim_point

    @property
    def lowest_z(self) -> float:
        lowest = min(float(np.asarray(tip)[2]) for tip in self.tips.values())
        if self.held is not None:
            lowest = min(lowest, float(self.held.lower[2]))
        return lowest


def approach_goal(hand: HandState, target: TargetBox, clearance_m: float = PREGRASP_CLEARANCE_M) -> np.ndarray:
    """End-effector position that puts the hand's grasp space above ``target`` (orientation kept)."""

    reference = hand.reference_point
    shift = np.array([
        target.center[0] - reference[0],
        target.center[1] - reference[1],
        target.top + float(clearance_m) - hand.lowest_z,
    ])
    return np.asarray(hand.ee_pos, dtype=np.float64) + shift


def _segment_passes_over(start_xy: np.ndarray, end_xy: np.ndarray, box: TargetBox, margin_m: float) -> bool:
    for fraction in np.linspace(0.0, 1.0, 41):
        point = start_xy + fraction * (end_xy - start_xy)
        if np.all(point >= box.lower[:2] - margin_m) and np.all(point <= box.upper[:2] + margin_m):
            return True
    return False


def _needed_rise(hand: HandState, goal: np.ndarray, obstacles: Sequence[TargetBox], margin_m: float) -> float:
    """How far the hand must rise (negative: may sink) so its lowest point clears what it passes over.

    Clears by ``margin_m`` every obstacle box that the hand (its fingertips
    and grasp point, which keep their offsets from the end effector on these
    paths) passes over on the way from here to above the goal.
    """

    start = np.asarray(hand.ee_pos, dtype=np.float64)
    offsets = [np.asarray(p, dtype=np.float64)[:2] - start[:2]
               for p in [hand.reference_point, *hand.tips.values()]]
    tops = [box.top for box in obstacles
            if any(_segment_passes_over(start[:2] + offset, goal[:2] + offset, box, 0.05) for offset in offsets)]
    return (max(tops) + margin_m - hand.lowest_z) if tops else 0.0


def _unique(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = [np.asarray(points[0], dtype=np.float64)]
    for point in points[1:]:
        if np.linalg.norm(np.asarray(point) - unique[-1]) > 1e-6:
            unique.append(np.asarray(point, dtype=np.float64))
    return unique


def straight_waypoints(hand: HandState, goal_pos: np.ndarray, obstacles: Sequence[TargetBox],
                       margin_m: float = STRAIGHT_LIFT_MARGIN_M) -> list[np.ndarray]:
    """Lift, traverse, descend: end-effector positions from the current pose to ``goal_pos``.

    The traverse height keeps the hand's lowest point ``margin_m`` above every
    obstacle box that the hand passes over on the way, and never goes below
    where the hand already is. No collision check.
    """

    start = np.asarray(hand.ee_pos, dtype=np.float64)
    goal = np.asarray(goal_pos, dtype=np.float64)
    travel_z = max(start[2] + max(0.0, _needed_rise(hand, goal, obstacles, margin_m)), goal[2])
    return _unique([start, np.array([start[0], start[1], travel_z]), np.array([goal[0], goal[1], travel_z]), goal])


def diagonal_waypoints(hand: HandState, goal_pos: np.ndarray, obstacles: Sequence[TargetBox],
                       margin_m: float = STRAIGHT_LIFT_MARGIN_M) -> list[np.ndarray]:
    """One straight line to above the goal at the clearing height, then descend (no lift first).

    When the hand starts high, this sinks on the way instead of travelling at
    the start height: near the robot base a high traverse folds the elbow to
    its limit (2026-11-06 forward-mount check: Panda joint 4 at -3.055 of
    -3.072 rad above the butter, the descent then had no solution). From a
    low start it rises on the way; the worker's clearance check decides.
    """

    start = np.asarray(hand.ee_pos, dtype=np.float64)
    goal = np.asarray(goal_pos, dtype=np.float64)
    travel_z = max(start[2] + _needed_rise(hand, goal, obstacles, margin_m), goal[2])
    return _unique([start, np.array([goal[0], goal[1], travel_z]), goal])


def cartesian_candidates(hand: HandState, goal_pos: np.ndarray, obstacles: Sequence[TargetBox],
                         lift_margins_m: Sequence[float] = CARTESIAN_LIFT_MARGINS_M,
                         direct_within_m: float = DIRECT_PATH_WITHIN_M) -> list[list[np.ndarray]]:
    """Routes the checked planner tries in order.

    A direct line when the goal is close, then lift / traverse / descend at
    rising margins, then the diagonal routes at the same margins.
    """

    start = np.asarray(hand.ee_pos, dtype=np.float64)
    goal = np.asarray(goal_pos, dtype=np.float64)
    routes: list[list[np.ndarray]] = []

    def add(points: list[np.ndarray]) -> None:
        if not any(len(points) == len(route) and all(np.allclose(a, b, atol=1e-9) for a, b in zip(points, route))
                   for route in routes):
            routes.append(points)

    if float(np.linalg.norm(goal - start)) <= direct_within_m:
        add([start, goal])
    for margin in lift_margins_m:
        add(straight_waypoints(hand, goal, obstacles, margin))
    for margin in lift_margins_m:
        add(diagonal_waypoints(hand, goal, obstacles, margin))
    return routes


def _min_jerk(u: float) -> float:
    return 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5


# ------------------------------------------------------------- motion path
class PlannedMotion:
    """Follow end-effector waypoints on a minimum-jerk time law (smooth start and stop).

    ``segmented`` (the lift / traverse / descend routes): every segment gets
    its own minimum-jerk law, so the hand comes to rest at each corner and the
    arm follows the checked straight lines instead of cutting the corners
    (the total time is the same as one law over the whole path). Otherwise
    one law runs over the whole polyline (the dense OMPL paths). Peak linear
    / angular speeds stay within the limits. Time is simulated time, so
    ``--sim-speed`` slows the motion in real time as well.
    """

    def __init__(self, positions: Sequence[np.ndarray], rotations: Sequence[np.ndarray],
                 max_speed_m_s: float = PLANNER_MAX_SPEED_M_S,
                 max_angular_speed_rad_s: float = PLANNER_MAX_ANGULAR_SPEED_RAD_S,
                 segmented: bool = False) -> None:
        if len(positions) != len(rotations) or len(positions) < 1:
            raise ValueError("need one rotation per waypoint and at least one waypoint")
        if max_speed_m_s <= 0.0 or max_angular_speed_rad_s <= 0.0:
            raise ValueError("speeds must be positive")
        self.positions = [np.asarray(p, dtype=np.float64).reshape(3) for p in positions]
        self.quats = [mat_to_quat(r) for r in rotations]
        for index in range(1, len(self.quats)):  # keep consecutive quaternions in one hemisphere
            if np.dot(self.quats[index - 1], self.quats[index]) < 0.0:
                self.quats[index] = -self.quats[index]
        # Drop repeated waypoints (zero-length, zero-angle segments).
        keep = [0] + [i for i in range(1, len(self.positions))
                      if np.linalg.norm(self.positions[i] - self.positions[i - 1]) > 1e-9
                      or rotation_angle(quat_to_mat(self.quats[i - 1]), quat_to_mat(self.quats[i])) > 1e-9]
        self.positions = [self.positions[i] for i in keep]
        self.quats = [self.quats[i] for i in keep]
        lengths = [float(np.linalg.norm(b - a)) for a, b in zip(self.positions, self.positions[1:])]
        angles = [rotation_angle(quat_to_mat(a), quat_to_mat(b)) for a, b in zip(self.quats, self.quats[1:])]
        # Arc parameter: per segment the larger of the translation / rotation times.
        self._segment = np.array([max(l / max_speed_m_s, a / max_angular_speed_rad_s)
                                  for l, a in zip(lengths, angles)], dtype=np.float64)
        self._cumulative = np.concatenate(([0.0], np.cumsum(self._segment)))
        self.segmented = bool(segmented)
        self._starts = _MIN_JERK_PEAK_RATIO * self._cumulative  # segmented: each segment's start time
        self.nominal_s = float(self._cumulative[-1])
        self.duration_s = _MIN_JERK_PEAK_RATIO * self.nominal_s
        self.path_length_m = float(sum(lengths))
        self.rotation_rad = float(sum(angles))
        self.elapsed_s = 0.0

    @property
    def final(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions[-1].copy(), quat_to_mat(self.quats[-1])

    @property
    def finished(self) -> bool:
        return self.elapsed_s >= self.duration_s

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.duration_s <= 0.0 or len(self.positions) == 1:
            return self.final
        if self.segmented:
            t = float(np.clip(time_s, 0.0, self.duration_s))
            index = int(np.clip(np.searchsorted(self._starts, t, side="right") - 1, 0, len(self._segment) - 1))
            span = _MIN_JERK_PEAK_RATIO * self._segment[index]
            fraction = 1.0 if span <= 0.0 else _min_jerk(float(np.clip((t - self._starts[index]) / span, 0.0, 1.0)))
        else:
            u = float(np.clip(time_s / self.duration_s, 0.0, 1.0))
            progress = _min_jerk(u) * self.nominal_s
            index = int(np.clip(np.searchsorted(self._cumulative, progress, side="right") - 1, 0, len(self._segment) - 1))
            span = self._segment[index]
            fraction = 1.0 if span <= 0.0 else float(np.clip((progress - self._cumulative[index]) / span, 0.0, 1.0))
        position = self.positions[index] + fraction * (self.positions[index + 1] - self.positions[index])
        return position, quat_to_mat(slerp(self.quats[index], self.quats[index + 1], fraction))

    def advance(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        self.elapsed_s = min(self.duration_s, self.elapsed_s + max(0.0, float(dt_s)))
        return self.sample(self.elapsed_s)


# ---------------------------------------------------------- scene readers
def _subtree_bodies(model: Any, root_body: int) -> set[int]:
    bodies = {int(root_body)}
    changed = True
    while changed:
        changed = False
        for body in range(int(model.nbody)):
            if body not in bodies and int(model.body_parentid[body]) in bodies and body != 0:
                bodies.add(body)
                changed = True
    return bodies


def _collision_geoms(model: Any, bodies: set[int]) -> list[int]:
    return [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in bodies
            and (int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0)]


def geom_world_box(model: Any, data: Any, geoms: Sequence[int]) -> Optional[TargetBox]:
    """World AABB of the geoms' local bounding boxes (``geom_aabb``: centre, half sizes)."""

    corners = []
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    for g in geoms:
        aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
        local = aabb[:3] + signs * aabb[3:]
        rotation = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        corners.append(np.asarray(data.geom_xpos[g], dtype=np.float64) + local @ rotation.T)
    if not corners:
        return None
    stacked = np.concatenate(corners)
    return TargetBox(stacked.min(axis=0), stacked.max(axis=0))


@dataclass
class SceneTarget:
    number: int
    name: str
    kind: str  # "object" or "fixture"
    root_body: int = -1
    bodies: set[int] = field(default_factory=set)
    geoms: list[int] = field(default_factory=list)


def list_targets(env: Any, limit: int = 9) -> list[SceneTarget]:
    """Task objects, then fixtures, numbered 1..9 in the task's own (BDDL) order."""

    import mujoco

    model = env.sim.model._model
    found: list[SceneTarget] = []
    groups = (("object", getattr(env, "objects_dict", {}) or {}), ("fixture", getattr(env, "fixtures_dict", {}) or {}))
    for kind, entries in groups:
        for name, entry in entries.items():
            root = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entry.root_body))
            if root < 0:
                continue
            bodies = _subtree_bodies(model, root)
            geoms = _collision_geoms(model, bodies)
            if geoms:
                found.append(SceneTarget(len(found) + 1, str(name), kind, root, bodies, geoms))
    return found[:limit]


def robot_geoms(env: Any) -> set[int]:
    """Every geom under the robot's root body (arm, mount and hand)."""

    import mujoco

    model = env.sim.model._model
    base = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base"))
    if base < 0:
        raise RuntimeError("robot0_base body not found")
    root = base
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    bodies = _subtree_bodies(model, root)
    return {g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in bodies}


class FingertipLocator:
    """World fingertip points of the Shadow hand (end of each distal collision mesh)."""

    def __init__(self, env: Any, prefix: str = "gripper0_rh_") -> None:
        import mujoco

        model = env.sim.model._model
        self._bodies: dict[str, int] = {}
        self._local: dict[str, np.ndarray] = {}
        for finger in FINGERS:
            body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "{}{}distal".format(prefix, finger)))
            if body < 0:
                raise RuntimeError("Shadow distal body not found for {}".format(finger))
            geoms = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == body
                     and int(model.geom_contype[g]) != 0]
            if not geoms:
                raise RuntimeError("Shadow distal collision geom not found for {}".format(finger))
            g = geoms[0]
            aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
            geom_rotation = np.empty(9)
            mujoco.mju_quat2Mat(geom_rotation, np.asarray(model.geom_quat[g], dtype=np.float64))
            geom_rotation = geom_rotation.reshape(3, 3)
            candidates = []
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    face = aabb[:3].copy()
                    face[axis] += sign * aabb[3 + axis]
                    candidates.append(np.asarray(model.geom_pos[g], dtype=np.float64) + geom_rotation @ face)
            # The face farthest from the distal joint (the body origin) is the tip.
            self._local[finger] = max(candidates, key=lambda p: float(np.linalg.norm(p)))
            self._bodies[finger] = body

    def tips(self, data: Any) -> dict[str, np.ndarray]:
        out = {}
        for finger, body in self._bodies.items():
            rotation = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
            out[finger] = np.asarray(data.xpos[body], dtype=np.float64) + rotation @ self._local[finger]
        return out


class FingerRootLocator:
    """World finger-root points (``ROOT_FINGERS``) and palm normal of the Shadow hand."""

    def __init__(self, env: Any, prefix: str = "gripper0_rh_") -> None:
        import mujoco

        from .shadow_gripper import SHADOW_PALMAR_NORMAL_LOCAL

        model = env.sim.model._model
        self._bodies: dict[str, int] = {}
        for finger in ROOT_FINGERS:
            # The proximal phalanx origin is the finger's MCP flexion joint (J3).
            body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "{}{}proximal".format(prefix, finger)))
            if body < 0:
                raise RuntimeError("Shadow proximal body not found for {}".format(finger))
            self._bodies[finger] = body
        self._palm = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "palm"))
        if self._palm < 0:
            raise RuntimeError("Shadow palm body not found")
        self._normal_local = np.asarray(SHADOW_PALMAR_NORMAL_LOCAL, dtype=np.float64)

    def roots(self, data: Any) -> dict[str, np.ndarray]:
        return {finger: np.array(data.xpos[body], dtype=np.float64) for finger, body in self._bodies.items()}

    def palm_normal(self, data: Any) -> np.ndarray:
        return np.asarray(data.xmat[self._palm], dtype=np.float64).reshape(3, 3) @ self._normal_local


def held_targets(model: Any, data: Any, targets: Sequence[SceneTarget], hand_geoms: set[int],
                 hand_bodies_for: Callable[[int], int]) -> list[SceneTarget]:
    """Objects touched by at least two distinct hand bodies (an opposed grasp)."""

    touching: dict[int, set[int]] = {}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        for hand_geom, other in ((g1, g2), (g2, g1)):
            if hand_geom in hand_geoms:
                for target in targets:
                    if target.kind == "object" and other in target.geoms:
                        touching.setdefault(target.number, set()).add(hand_bodies_for(hand_geom))
    return [t for t in targets if len(touching.get(t.number, ())) >= 2]


# ------------------------------------------------------------------- OMPL
def discover_planner_python(explicit: Optional[str] = None) -> Optional[Path]:
    """``--planner-python``, then ``OMPL_PLANNER_PYTHON``, then the ``ompl_planner`` sibling Conda env."""

    if explicit:
        return Path(explicit).expanduser()
    configured = os.environ.get("OMPL_PLANNER_PYTHON")
    if configured:
        return Path(configured).expanduser()
    # The mainline venv records its base Conda env; ompl_planner is its sibling.
    cfg = ROOT / ".venv-robosuite" / "pyvenv.cfg"
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("home"):
                home = Path(line.split("=", 1)[1].strip())
                candidate = home.parent / OMPL_ENV_NAME / "python.exe"
                if candidate.is_file():
                    return candidate
    return None


class PlanningFailed(RuntimeError):
    """A plan that cannot be made (the message says why, for the operator).

    ``reason``: unreachable (no arm posture holds the hand there with this
    orientation), blocked (too close to something), no_path (search ran out).
    """

    def __init__(self, message: str, reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason


class WorkerError(RuntimeError):
    """The worker answered a request with an error; ``reason`` is set for planning failures."""

    def __init__(self, op: str, error: str, reason: Optional[str] = None) -> None:
        super().__init__("OMPL planner {} failed: {}".format(op, error))
        self.error, self.reason = error, reason


class OmplPlannerSidecar:
    """JSON-lines child process running OMPL + MuJoCo collision checks (one request at a time)."""

    def __init__(self, python_path: Path, worker: Path = OMPL_WORKER) -> None:
        python_path = Path(python_path)
        if not python_path.is_file():
            raise RuntimeError("OMPL planner Python not found: {}".format(python_path))
        environment = os.environ.copy()
        env_root = python_path.parent
        environment["PATH"] = os.pathsep.join(
            [str(env_root / "Library" / "bin"), str(env_root / "Scripts"), str(env_root), environment.get("PATH", "")])
        environment.pop("PYTHONPATH", None)
        self._process = subprocess.Popen(
            [str(python_path), "-u", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._responses: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stderr: list[str] = []
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False
        threading.Thread(target=self._read_stdout, name="ompl-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="ompl-stderr", daemon=True).start()

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                self._stderr.append("stdout: " + text)
                continue
            if isinstance(response, dict):
                self._responses.put(response)

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            if line.strip():
                self._stderr = (self._stderr + [line.strip()])[-20:]

    def request(self, op: str, timeout_s: float, **payload: Any) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError("OMPL planner is not running: {}".format(" | ".join(self._stderr[-5:])))
            self._next_id += 1
            request_id = self._next_id
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps({"id": request_id, "op": op, **payload}) + "\n")
            self._process.stdin.flush()
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    response = self._responses.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty:
                    raise RuntimeError("OMPL planner timed out after {:.1f} s during {}".format(timeout_s, op))
                if response.get("id") == request_id:
                    break
            if not response.get("ok"):
                raise WorkerError(op, str(response.get("error")), response.get("reason"))
            return response.get("result", {})

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            if self._process.poll() is None and self._process.stdin is not None:
                self._process.stdin.write(json.dumps({"id": 0, "op": "close"}) + "\n")
                self._process.stdin.flush()
                self._process.stdin.close()
            self._process.wait(timeout=3.0)
        except Exception:
            self._process.kill()


# ------------------------------------------------------- planner controller
@dataclass
class PlanRequest:
    request_id: int
    target: SceneTarget
    goal_pos: np.ndarray
    goal_rot: np.ndarray
    hand: HandState
    held: list[str]
    requested_at: float
    # Everything the plan needs, captured on the control thread (the worker
    # thread never reads the live simulator).
    payload: dict[str, Any] = field(default_factory=dict)


class ApproachPlanner:
    """State machine: idle -> planning -> moving -> idle. Any operator input cancels it.

    ``tick`` returns the end-effector goal for this control step while the
    planner owns the arm, else ``None`` (the operator's SpaceMouse owns it).
    ``plan_fn(request)`` returns ``(positions, rotations, info)`` and runs on a
    worker thread so the control loop keeps ticking while OMPL searches.
    """

    def __init__(self, plan_fn: Callable[[PlanRequest], tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]],
                 backend: str, max_speed_m_s: float = PLANNER_MAX_SPEED_M_S,
                 max_angular_speed_rad_s: Optional[float] = None) -> None:
        self.plan_fn, self.backend = plan_fn, backend
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_angular_speed_rad_s = float(max_angular_speed_rad_s if max_angular_speed_rad_s is not None
                                             else max_speed_m_s * PLANNER_ANGULAR_PER_LINEAR)
        self.state = "idle"
        self.request: Optional[PlanRequest] = None
        self.motion: Optional[PlannedMotion] = None
        self.events: list[dict[str, Any]] = []
        self._results: "queue.Queue[tuple[int, Any]]" = queue.Queue()
        self._next_id = 0
        self._settle_s = 0.0

    @property
    def active(self) -> bool:
        return self.state in ("planning", "moving")

    def status(self) -> str:
        if self.request is None or self.state == "idle":
            return "manual"
        return "{} {} {}".format(self.state, self.request.target.number, self.request.target.name)

    def start(self, target: SceneTarget, goal_pos: np.ndarray, goal_rot: np.ndarray, hand: HandState,
              held: Sequence[str], payload: Optional[dict[str, Any]] = None) -> None:
        if self.active:
            self.cancel("new target")
        self._next_id += 1
        self.request = PlanRequest(self._next_id, target, np.asarray(goal_pos, dtype=np.float64),
                                   np.asarray(goal_rot, dtype=np.float64), hand, list(held), time.perf_counter(),
                                   dict(payload or {}))
        self.state, self.motion = "planning", None
        self.events.append({"event": "planner_request", "request_id": self._next_id, "backend": self.backend,
                            "target": target.name, "number": target.number, "kind": target.kind,
                            "held": list(held), "goal_ee_pos": self.request.goal_pos,
                            "start_ee_pos": hand.ee_pos, "aim": hand.aim})
        request = self.request
        threading.Thread(target=self._run_plan, args=(request,), name="approach-plan", daemon=True).start()

    def _run_plan(self, request: PlanRequest) -> None:
        try:
            self._results.put((request.request_id, self.plan_fn(request)))
        except Exception as exc:  # reported on the control thread
            self._results.put((request.request_id, exc))

    def cancel(self, reason: str) -> bool:
        if not self.active:
            return False
        self.events.append({"event": "planner_cancel", "reason": reason, "state": self.state,
                            "request_id": self.request.request_id if self.request else None})
        self.state, self.motion = "idle", None
        return True

    def tick(self, dt_sim_s: float, ee_pos: np.ndarray, ee_rot: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        while True:
            try:
                request_id, result = self._results.get_nowait()
            except queue.Empty:
                break
            if self.state != "planning" or self.request is None or request_id != self.request.request_id:
                continue  # a cancelled or superseded request
            if isinstance(result, Exception):
                self.events.append({"event": "planner_failed", "request_id": request_id,
                                    "error": str(result) if isinstance(result, PlanningFailed)
                                    else "{}: {}".format(type(result).__name__, result),
                                    "reason": getattr(result, "reason", None)})
                self.state = "idle"
                continue
            positions, rotations, info = result
            # Start from where the arm is now (the plan began from the same pose).
            positions = [np.asarray(ee_pos, dtype=np.float64)] + list(positions)
            rotations = [np.asarray(ee_rot, dtype=np.float64)] + list(rotations)
            self.motion = PlannedMotion(positions, rotations, self.max_speed_m_s, self.max_angular_speed_rad_s,
                                        segmented=bool(info.get("segmented")))
            self._settle_s = 0.0
            self.state = "moving"
            self.events.append({"event": "planner_result", "request_id": request_id,
                                "plan_ms": 1e3 * (time.perf_counter() - self.request.requested_at),
                                "waypoints": len(positions), "path_length_m": self.motion.path_length_m,
                                "path_rotation_rad": self.motion.rotation_rad,
                                "duration_sim_s": self.motion.duration_s, **info})
        if self.state != "moving" or self.motion is None:
            return None
        goal = self.motion.advance(dt_sim_s)
        if self.motion.finished:
            final_pos, final_rot = self.motion.final
            self._settle_s += dt_sim_s
            close = (np.linalg.norm(np.asarray(ee_pos) - final_pos) < ARRIVAL_TOLERANCE_M
                     and rotation_angle(ee_rot, final_rot) < ARRIVAL_TOLERANCE_RAD)
            if close or self._settle_s >= ARRIVAL_TIMEOUT_S:
                self.state = "idle"
                self.events.append({"event": "planner_arrived", "request_id": self.request.request_id if self.request else None,
                                    "within_tolerance": bool(close),
                                    "ee_error_m": float(np.linalg.norm(np.asarray(ee_pos) - final_pos))})
        return goal

    def drain_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events
