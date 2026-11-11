"""LIBERO Panda + Shadow teleop with a motion planner for the approach (this week's experiment).

Copied from ``run_libero_spatial_teleop`` (which stays unchanged) and extended:

- ``--sim-speed``: slow motion. The control loop still ticks at 20 Hz wall
  time (views and camera hand too), but each tick advances the simulation by
  ``sim_speed x 50 ms`` (robosuite ``control_freq = 20 / sim_speed``; the
  2 ms physics step is unchanged). SpaceMouse increments per tick are scaled
  by the same factor, so the arm keeps its speed in simulated time.
- Number keys 1-9 pick a task object / fixture (numbers drawn in the views
  and printed at start). The planner moves the end effector until the hand's
  grasp space is above it (see ``libero_motion_planning``). While planning or
  moving, any SpaceMouse input stops it on the spot and the SpaceMouse has
  the arm again; ``0`` cancels too. On arrival the SpaceMouse takes over for
  the fine approach. The camera hand drives the fingers throughout.
- Five red rays, one per fingertip, straight down (world -z); ``L`` toggles.
  The palm laser of the original entry is not drawn here.
- The target numbers are small badges drawn on the image (no names).
- ``M`` switches the hand mount between "down" (fingers point down) and
  "forward" (palm parallel to the table) by rebuilding the task with the
  other mount; like ``R`` it restarts the task from its initial state. (Not
  ``C``: the camera inset of the shared hand input prints "C re-centre dex".)

Scope: libero_spatial and libero_object. Run only with ``.venv-libero``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
from typing import Any, Optional, Sequence

import numpy as np

from .config import WristConfig
from .device import SpaceMouseHid
from .hand_input import CameraHandWorker
from .libero_motion_planning import (
    FINGERS,
    PLANNER_BACKENDS,
    PLANNER_MAX_SPEED_M_S,
    PLANNER_SPEED_RANGE_M_S,
    PREGRASP_CLEARANCE_M,
    ApproachPlanner,
    FingerRootLocator,
    FingertipLocator,
    HandState,
    OmplPlannerSidecar,
    PlanningFailed,
    PlanRequest,
    SceneTarget,
    TargetBox,
    WorkerError,
    approach_goal,
    cartesian_candidates,
    discover_planner_python,
    geom_world_box,
    held_targets,
    list_targets,
    robot_geoms,
    straight_waypoints,
)
from .libero_spatial_teleop import (
    DETAIL_FOLLOW,
    LIBERO_ARM_ROTATION_SPEED_SCALE,
    LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    LIBERO_GOAL_LEASH_M,
    LIBERO_GOAL_LEASH_RAD,
    SCENE_FOLLOW,
    LiberoActionComposer,
    LiberoTeleopSpec,
    OfficialLiberoSpatialSession,
    _NeutralInput,
    _RunLog,
    _WindowsTimerResolution,
    _shadow_tri_view_config,
    describe_object_masses,
    follow_robot_base,
    shadow_spec,
)
from .libero_views import LiberoViewWindow, Panel, default_view_size, enable_dpi_awareness
from .loop_timing import FixedDelayResampler, FixedRateClock
from .mapper import SixDofMapper
from .runtime_controls import KeyboardCommandRouter, ResetFeedback
from .shadow_dex import ShadowDexHandInput, ShadowDexRetargeter
from .shadow_gripper import SHADOW_MOUNTS
from .shadow_hand_input import ShadowHandGestureController, ShadowLegacyHandInput

# Simulated seconds per real second. Each tick's simulated time must be a
# whole number of 2 ms physics steps: 50 / 40 / 30 / 20 ms.
SIM_SPEED_CHOICES = (1.0, 0.8, 0.6, 0.4)
PLANNER_SUITES = ("libero_spatial", "libero_object")
# RRT-Connect is only the fallback now (straight routes are checked first).
OMPL_TIME_LIMIT_S = 2.0
OMPL_SIMPLIFY_S = 0.5
RAY_RGBA = (1.0, 0.0, 0.0, 0.9)
RAY_RADIUS_M = 0.0015
RAY_DOT_RADIUS_M = 0.004
RAY_MAX_LENGTH_M = 1.5
# Target numbers: small badges drawn on the image (RGB), above each target.
LABEL_RGB = (255, 216, 26)
LABEL_LIFT_M = 0.03
LABEL_SLOTS = ("front", "side")
BADGE_RADIUS_PX = 8.0
BADGE_FONT_SCALE = 0.36
DIGIT_REPEAT_S = 0.3
# M: switch the hand mount (rebuilds the task with the other one, a few seconds).
MOUNT_SWITCH_KEY = "m"
MOUNT_SWITCH_REPEAT_S = 1.0
MOUNT_TEXT = {"down": "fingers down", "forward": "palm parallel to the table"}
MOUNT_SWITCH_NOTICE = "SWITCHING HAND MOUNT - please wait\nR is ignored until the scene is back"
# The compiled scene for the planner worker (~250 MB): one temp file per run, removed on exit.
SCENE_DIR = Path(tempfile.gettempdir()) / "libero_planner_scenes"
SCENE_STALE_S = 12 * 3600.0


def mount_of(spec: LiberoTeleopSpec) -> str:
    """Which of ``SHADOW_MOUNTS`` a Shadow spec is."""

    for mount in SHADOW_MOUNTS:
        if shadow_spec(mount) == spec:
            return mount
    raise ValueError("not a Panda + Shadow spec: {}".format(spec.name))


def other_mount(mount: str) -> str:
    return "forward" if mount == "down" else "down"


def _remove_quietly(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:  # already gone, or still open in the worker (removed at exit then)
        pass


def _remove_stale_scenes(now: float) -> None:
    """Scene files of earlier runs that ended without cleaning up (crash, kill)."""

    try:
        for path in SCENE_DIR.glob("scene_*.mjb"):
            try:
                if now - path.stat().st_mtime > SCENE_STALE_S:
                    path.unlink()
            except OSError:
                pass
    except OSError:
        pass


def env_config_for_sim_speed(config: WristConfig, sim_speed: float) -> WristConfig:
    """The config the LIBERO session builds its env from: control_freq = control_hz / sim_speed."""

    if float(sim_speed) not in SIM_SPEED_CHOICES:
        raise ValueError("sim speed must be one of {}".format(", ".join("{:g}".format(v) for v in SIM_SPEED_CHOICES)))
    return replace(config, control_hz=config.control_hz / float(sim_speed))


def arm_config_for_speeds(config: WristConfig, arm_speed: float, rotation_speed: float, sim_speed: float) -> WristConfig:
    """Per-tick SpaceMouse increments: the entry's speed scales, times the sim speed."""

    for name, value in (("arm speed", arm_speed), ("rotation speed", rotation_speed)):
        if not np.isfinite(value) or not 0.1 <= float(value) <= 5.0:
            raise ValueError("{} scale must be in [0.1, 5]".format(name))
    translation, rotation = float(arm_speed) * float(sim_speed), float(rotation_speed) * float(sim_speed)
    return replace(
        config,
        translation_scale_m=config.translation_scale_m * translation,
        max_translation_delta_m=config.max_translation_delta_m * translation,
        rotation_scale_rad=config.rotation_scale_rad * rotation,
        max_rotation_delta_rad=config.max_rotation_delta_rad * rotation,
    )


# ------------------------------------------------------------ composer
class PlannerActionComposer(LiberoActionComposer):
    """The original composer plus direct goal setting for planner-owned ticks."""

    def set_goal(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self.goal_pos = np.asarray(position, dtype=np.float64).copy()
        self.goal_ori = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
        self._moving = False

    def hold_here(self) -> None:
        """Stop on the spot: the desired pose becomes the measured pose."""
        self.goal_pos, self.goal_ori = self.ee_pose()
        self._moving = False


# ------------------------------------------------------- view decorations
@dataclass(frozen=True)
class ExtraGeom:
    slots: tuple[str, ...]
    kind: str  # "sphere" or "capsule"
    start: tuple[float, float, float]
    end: Optional[tuple[float, float, float]] = None
    size: float = 0.004
    rgba: tuple[float, float, float, float] = RAY_RGBA
    label: str = ""


def _append_geom(scene: Any, spec: ExtraGeom) -> bool:
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    rgba = np.asarray(spec.rgba, dtype=np.float32)
    if spec.kind == "sphere":
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([spec.size, 0.0, 0.0]),
                            np.asarray(spec.start, dtype=np.float64), np.eye(3).ravel(), rgba)
    else:
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, spec.size,
                             np.asarray(spec.start, dtype=np.float64), np.asarray(spec.end, dtype=np.float64))
    geom.label = spec.label
    scene.ngeom += 1
    return True


def project_point(glcam: Any, point: Sequence[float], width: int, height: int) -> Optional[tuple[float, float]]:
    """Pixel (u, v) of a world point in a panel drawn with this mjvGLCamera, or None behind it.

    The same frustum ``mjr_render`` sets up (half-width from the viewport
    aspect when ``frustum_width`` is 0); within 1-3 px of the rendered image.
    """

    position = np.asarray(glcam.pos, dtype=np.float64)
    forward = np.asarray(glcam.forward, dtype=np.float64)
    up = np.asarray(glcam.up, dtype=np.float64)
    offset = np.asarray(point, dtype=np.float64) - position
    depth = float(offset @ forward)
    near = float(glcam.frustum_near)
    if depth <= near or width <= 0 or height <= 0:
        return None
    half_h = 0.5 * (float(glcam.frustum_top) - float(glcam.frustum_bottom))
    mid_v = 0.5 * (float(glcam.frustum_top) + float(glcam.frustum_bottom))
    half_w = float(glcam.frustum_width) if float(glcam.frustum_width) > 0.0 else half_h * width / height
    x = near * float(offset @ np.cross(forward, up)) / depth
    y = near * float(offset @ up) / depth
    return ((0.5 + 0.5 * (x - float(glcam.frustum_center)) / half_w) * width,
            (0.5 - 0.5 * (y - mid_v) / half_h) * height)


class _AnnotatedStatus(str):
    """The status line plus the target badges (frame x, y, text) captured with the same scenes."""

    badges: tuple[tuple[int, int, str], ...] = ()

    def __new__(cls, text: str, badges: Sequence[tuple[int, int, str]] = ()) -> "_AnnotatedStatus":
        status = super().__new__(cls, text)
        status.badges = tuple(badges)
        return status


def _draw_badge(cv2: Any, frame: np.ndarray, x: int, y: int, text: str, ui_scale: float) -> None:
    radius = max(5, int(round(BADGE_RADIUS_PX * ui_scale)))
    scale = BADGE_FONT_SCALE * max(ui_scale, 0.6)
    cv2.circle(frame, (x, y), radius, LABEL_RGB, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)
    (width, height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(frame, text, (x - width // 2, y + height // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1,
                cv2.LINE_AA)


class PlannerViewWindow(LiberoViewWindow):
    """The LIBERO view window plus rays (scene geoms) and target number badges (drawn on the image)."""

    def __init__(self, *args: Any, scene_extras: Any = None, badge_points: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scene_extras = scene_extras
        self.badge_points = badge_points

    def _decorate(self, cv2: Any, frame: np.ndarray, status_text: Optional[str]) -> None:
        super()._decorate(cv2, frame, status_text)
        for x, y, text in getattr(status_text, "badges", ()):
            _draw_badge(cv2, frame, x, y, text, self.ui_scale)

    def _badges(self, slot: str, scene: Any, points: Sequence[tuple[str, np.ndarray]]) -> list[tuple[int, int, str]]:
        import mujoco

        if slot not in LABEL_SLOTS or not points:
            return []
        x0, y0, width, height = self.rects[slot]
        glcam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        out = []
        for text, point in points:
            pixel = project_point(glcam, point, width, height)
            if pixel is not None and 0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height:
                out.append((int(round(x0 + pixel[0])), int(round(y0 + pixel[1])), text))
        return out

    def _capture(self, status_text: Optional[str]) -> bool:  # copy of the base method + extras
        import mujoco

        started = time.perf_counter()
        with self._lock:
            if not self._sets or self._closed:
                return False
            index = 0 if self._rendering == 1 else 1 if self._rendering == 0 else (
                1 - self._ready if self._ready is not None else 0)
            if self._ready is not None:
                self._stats["dropped"] += 1
                if self._ready == index:
                    self._ready = None
            target = self._sets[index]
            generation = self._generation
        model, data = self._sim.model._model, self._sim.data._data
        sim_time = float(data.time)
        lookats = self._follow_lookats(sim_time) if self._site_id >= 0 else {}
        extras: Sequence[ExtraGeom] = ()
        points: Sequence[tuple[str, np.ndarray]] = ()
        badges: list[tuple[int, int, str]] = []
        try:
            if self.scene_extras is not None:
                extras = self.scene_extras()
            if self.badge_points is not None:
                points = self.badge_points()
        except Exception as exc:
            self._error = "scene extras: {}: {}".format(type(exc).__name__, exc)
        perturb = mujoco.MjvPerturb()
        global_fovy = float(model.vis.global_.fovy)
        try:
            for slot, panel in self.panels.items():
                camera = mujoco.MjvCamera()
                if panel.camera is not None:
                    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    camera.fixedcamid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, panel.camera))
                else:
                    follow = panel.follow
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = lookats[slot] + np.asarray(follow.lookat_offset_m)
                    camera.distance = follow.distance_m
                    camera.azimuth = follow.azimuth_deg + (self.orbit_deg if follow.orbit else 0.0)
                    camera.elevation = follow.elevation_deg
                    model.vis.global_.fovy = follow.fovy_deg
                mujoco.mjv_updateScene(model, data, self._vopt, perturb, camera,
                                       mujoco.mjtCatBit.mjCAT_ALL, target.scenes[slot])
                for spec in extras:
                    if slot in spec.slots:
                        _append_geom(target.scenes[slot], spec)
                badges.extend(self._badges(slot, target.scenes[slot], points))
                model.vis.global_.fovy = global_fovy
        finally:
            model.vis.global_.fovy = global_fovy
        target.generation, target.sim_time = generation, sim_time
        target.status_text = _AnnotatedStatus(status_text or "", badges)
        target.captured_at = time.perf_counter()
        with self._lock:
            if generation == self._generation:
                self._ready = index
                self._lock.notify_all()
        self._capture_ms = 1e3 * (time.perf_counter() - started)
        return True


class FingertipRays:
    """One vertical ray per fingertip to the first non-robot surface below it."""

    def __init__(self, env: Any, exclude_geoms: set[int], enabled: bool = True) -> None:
        self.env, self.exclude, self.enabled = env, set(exclude_geoms), bool(enabled)
        self.locator = FingertipLocator(env)

    def _hit(self, model: Any, data: Any, start: np.ndarray) -> Optional[np.ndarray]:
        import mujoco

        down = np.array([0.0, 0.0, -1.0])
        origin, travelled = start.copy(), 0.0
        for _ in range(64):
            geom = np.full(1, -1, dtype=np.int32)
            distance = float(mujoco.mj_ray(model, data, origin, down, None, 1, -1, geom))
            if distance < 0.0 or travelled + distance > RAY_MAX_LENGTH_M:
                return None
            if int(geom[0]) not in self.exclude:
                return origin + down * distance
            origin = origin + down * (distance + 1e-4)
            travelled += distance + 1e-4
        return None

    def segments(self) -> dict[str, tuple[np.ndarray, Optional[np.ndarray]]]:
        model, data = self.env.sim.model._model, self.env.sim.data._data
        return {finger: (tip, self._hit(model, data, tip)) for finger, tip in self.locator.tips(data).items()}

    def geoms(self) -> list[ExtraGeom]:
        if not self.enabled:
            return []
        out = []
        slots = ("front", "wrist", "side")
        for _finger, (tip, hit) in self.segments().items():
            end = hit if hit is not None else tip + np.array([0.0, 0.0, -RAY_MAX_LENGTH_M])
            out.append(ExtraGeom(slots, "capsule", tuple(tip), tuple(end), RAY_RADIUS_M))
            if hit is not None:
                out.append(ExtraGeom(slots, "sphere", tuple(hit), None, RAY_DOT_RADIUS_M))
        return out


# ------------------------------------------------------------ the scene
class PlannerScene:
    """Targets, robot / hand geoms and fingertips of the current model (rebuilt after a hard reset)."""

    def __init__(self, env: Any, composer: PlannerActionComposer) -> None:
        import mujoco

        self.env = env
        model = env.sim.model._model
        self.targets = list_targets(env)
        self.robot_geoms = robot_geoms(env)
        hand_bodies = {b for b in range(int(model.nbody))
                       if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("gripper0_")}
        self.hand_geoms = {g for g in self.robot_geoms if int(model.geom_bodyid[g]) in hand_bodies}
        self.fingertips = FingertipLocator(env)
        self.finger_roots = FingerRootLocator(env)
        robot = env.robots[0]
        self.arm_qpos_adr = [int(i) for i in robot._ref_joint_pos_indexes]
        self.arm_dof_adr = [int(i) for i in robot._ref_joint_vel_indexes]
        self.arm_joint_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) for name in robot.robot_joints]
        self.eef_site = composer.controller.eef_name
        self.composer = composer

    def by_number(self, number: int) -> Optional[SceneTarget]:
        return next((t for t in self.targets if t.number == number), None)

    def box(self, target: SceneTarget) -> TargetBox:
        box = geom_world_box(self.env.sim.model._model, self.env.sim.data._data, target.geoms)
        assert box is not None
        return box

    def held(self) -> list[SceneTarget]:
        model, data = self.env.sim.model._model, self.env.sim.data._data
        return held_targets(model, data, self.targets, self.hand_geoms, lambda g: int(model.geom_bodyid[g]))

    def hand_state(self, held: Sequence[SceneTarget]) -> HandState:
        ee_pos, ee_rot = self.composer.ee_pose()
        held_box = None
        if held:
            boxes = [self.box(t) for t in held]
            held_box = TargetBox(np.min([b.lower for b in boxes], axis=0), np.max([b.upper for b in boxes], axis=0))
        data = self.env.sim.data._data
        return HandState(ee_pos, ee_rot, self.fingertips.tips(data), held_box,
                         roots=self.finger_roots.roots(data), palm_normal=self.finger_roots.palm_normal(data))

    def badge_points(self) -> list[tuple[str, np.ndarray]]:
        """(number, world point a little above the target's top centre) per target."""
        out = []
        for target in self.targets:
            box = self.box(target)
            out.append((str(target.number), np.array([box.center[0], box.center[1], box.top + LABEL_LIFT_M])))
        return out

    def table(self) -> str:
        return "\n".join("  {}  {:32s} ({})".format(t.number, t.name, t.kind) for t in self.targets)


def _planner_view_window(tri_config: Any, env: Any, composer: LiberoActionComposer, *, headless: bool,
                         view_size: Optional[tuple[int, int]], key_handler: Any, on_close: Any, preview: Any,
                         scene_extras: Any, badge_points: Any = None) -> PlannerViewWindow:
    panels = (
        Panel("front", "", camera=tri_config.front_camera_name),
        Panel("wrist", "CLOSE-UP (follows hand)", follow=DETAIL_FOLLOW),
        Panel("side", "3D FROM FRONT-RIGHT (follows hand)", follow=SCENE_FOLLOW),
    )
    window = PlannerViewWindow(
        tri_config, panels, size=view_size or default_view_size(), show_window=not headless,
        track_site=composer.controller.eef_name, key_handler=key_handler, on_close=on_close,
        laser=None, preview=preview, scene_extras=scene_extras, badge_points=badge_points,
    )
    window.configure_cameras(env)
    return window


# ---------------------------------------------------------------- planner
class _OmplBridge:
    """Hands plan requests to the OMPL sidecar; (re)loads the scene file when it changed."""

    def __init__(self, sidecar: OmplPlannerSidecar) -> None:
        self.sidecar = sidecar
        self.loaded_version = -1

    def plan(self, request: PlanRequest) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        payload = dict(request.payload)
        scene = payload.pop("scene")
        if scene["version"] != self.loaded_version:
            self.sidecar.request("init", timeout_s=60.0, **scene["init"])
            self.loaded_version = scene["version"]
        try:
            result = self.sidecar.request("plan", timeout_s=OMPL_TIME_LIMIT_S + OMPL_SIMPLIFY_S + 30.0, **payload)
        except WorkerError as exc:
            if exc.reason is None:
                raise
            raise PlanningFailed(exc.error, exc.reason) from None
        positions = [np.asarray(p, dtype=np.float64) for p in result.pop("positions")]
        rotations = [np.asarray(r, dtype=np.float64).reshape(3, 3) for r in result.pop("rotations")]
        result.pop("joints", None)
        # End exactly on the checked goal (the requested one, or up to 4 cm
        # higher when the descent would touch what is below); the OMPL path's
        # last pose is IK-accurate to 2 mm / 0.01 rad only.
        positions[-1] = np.asarray(result["goal_pos"], dtype=np.float64)
        rotations[-1] = request.goal_rot.reshape(3, 3).copy()
        return positions, rotations, {"backend": "ompl", **result}


def _straight_plan(request: PlanRequest) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """The unchecked comparison: the first lift / traverse / descend route, same time law."""
    positions = [np.asarray(p, dtype=np.float64) for p in request.payload["waypoints"]]
    return positions, [request.goal_rot] * len(positions), {"backend": "straight", "strategy": "unchecked",
                                                            "segmented": True}


def _describe_plan(event: dict[str, Any]) -> str:
    """One console line for a planner_result event."""
    strategy = event.get("strategy")
    if strategy == "cartesian":
        how = "checked straight route {} of {}".format(event.get("route"), event.get("routes_tried"))
        if event.get("route_failures"):
            how = "checked straight route {} (earlier: {})".format(event.get("route"), "; ".join(event["route_failures"]))
    elif strategy == "ompl":
        how = "OMPL search, straight routes blocked"
    else:
        how = "unchecked straight route"
    line = "path found ({}): {:.2f} m, {:.0f} deg wrist turn, {:.1f} s simulated, {:.0f} ms planning".format(
        how, event["path_length_m"], np.degrees(event.get("path_rotation_rad", 0.0)), event["duration_sim_s"],
        event["plan_ms"])
    if event.get("closest_m") is not None:
        line += ", closest {:.0f} mm".format(1e3 * event["closest_m"])
    if event.get("goal_lift_m"):
        line += "; stops {:.0f} mm higher (the descent would touch what is below)".format(1e3 * event["goal_lift_m"])
    return line


def run_libero_planner_teleop(
    spec: LiberoTeleopSpec,
    config: WristConfig,
    *,
    task_id: int,
    task_order: int,
    init_state_id: int,
    seed: int,
    headless: bool,
    duration_s: Optional[float],
    camera_index: int,
    camera_display: bool,
    camera_mirror: bool,
    hand_control: str,
    synthetic_input: bool,
    suite: str = "libero_spatial",
    arm_speed_scale: float = LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    rotation_speed_scale: float = LIBERO_ARM_ROTATION_SPEED_SCALE,
    object_mass_scale: float = 1.0,
    object_masses_kg: Optional[dict[str, float]] = None,
    view_size: Optional[tuple[int, int]] = None,
    dex_grasp_terms: bool = True,
    thumb_lift_gain: Optional[float] = None,
    sim_speed: float = 1.0,
    planner_backend: str = "ompl",
    planner_python: Optional[str] = None,
    clearance_m: float = PREGRASP_CLEARANCE_M,
    rays_enabled: bool = True,
    smoke_target: Optional[int] = None,
    planner_speed_m_s: float = PLANNER_MAX_SPEED_M_S,
    smoke_keys: Sequence[tuple[float, str]] = (),
) -> int:
    """Run one LIBERO state with SpaceMouse arm, camera Shadow hand and the approach planner.

    ``smoke_target`` requests that target number once, ~0.5 s after start;
    ``smoke_keys`` presses keys at given run times, e.g. ``[(0.5, "3"),
    (4.0, "m")]`` (tests / headless smoke only).
    """

    if not spec.camera_hand:
        raise ValueError("the planner entry is for the Shadow hand")
    if hand_control not in ("dex", "legacy-curl", "off"):
        raise ValueError("hand control must be dex, legacy-curl, or off")
    if suite not in PLANNER_SUITES:
        raise ValueError("the planner entry covers {} only".format(" and ".join(PLANNER_SUITES)))
    if planner_backend not in PLANNER_BACKENDS:
        raise ValueError("planner must be one of {}".format(", ".join(PLANNER_BACKENDS)))
    if not np.isfinite(clearance_m) or not 0.0 <= clearance_m <= 0.2:
        raise ValueError("clearance must be within 0-0.2 m")
    low_speed, high_speed = PLANNER_SPEED_RANGE_M_S
    if not np.isfinite(planner_speed_m_s) or not low_speed <= planner_speed_m_s <= high_speed:
        raise ValueError("planner speed must be within {:g}-{:g} m/s".format(low_speed, high_speed))
    mount = mount_of(spec)
    env_config = env_config_for_sim_speed(config, sim_speed)
    config = arm_config_for_speeds(config, arm_speed_scale, rotation_speed_scale, sim_speed)
    enable_dpi_awareness()
    log = _RunLog(replace(spec, name=spec.name + "_planner"))
    mapper = SixDofMapper(config)
    device: Any = _NeutralInput() if synthetic_input else SpaceMouseHid(config)

    def make_session(hand_spec: LiberoTeleopSpec) -> OfficialLiberoSpatialSession:
        return OfficialLiberoSpatialSession(
            hand_spec, env_config, task_id=task_id, task_order=task_order, init_state_id=init_state_id,
            seed=seed, has_offscreen_renderer=True, suite=suite, object_mass_scale=object_mass_scale,
            object_masses_kg=object_masses_kg,
        )

    session = make_session(spec)
    keyboard = KeyboardCommandRouter()
    reset_feedback = ResetFeedback(keyboard)  # R during a reset / M rebuild is dropped; the views say so
    commands: "queue.Queue[tuple[str, int]]" = queue.Queue()
    last_digit: dict[int, float] = {}
    tri_view = camera = retargeter = sidecar = None
    composer = hand_input = gestures = rays = scene = None
    hand_resampler: Optional[FixedDelayResampler] = None
    hand_sample_seq = 0
    exit_code, reason = 0, "completed"
    last_buttons: tuple[bool, ...] = (False,) * 8
    screenshot_written = False
    loop_periods: list[float] = []
    clock = FixedRateClock(config.control_hz, high_resolution_timer=False)
    scene_version = 0
    scene_init: dict[str, Any] = {}
    scene_path: Optional[Path] = None
    planner_counts = {"requests": 0, "arrived": 0, "cancelled": 0, "failed": 0, "mount_switches": 0}

    def route(key: int, source: str) -> None:
        if tri_view is not None and key in (ord("v"), ord("V")):
            print("close-up view turned to {:.0f} deg".format(tri_view.cycle_detail_view()))
            return
        if ord("0") <= key <= ord("9"):
            now = time.monotonic()
            if now - last_digit.get(key, -1.0) < DIGIT_REPEAT_S:
                return
            last_digit[key] = now
            commands.put(("cancel", 0) if key == ord("0") else ("target", key - ord("0")))
            return
        if key in (ord("l"), ord("L")):
            now = time.monotonic()
            if now - last_digit.get(key, -1.0) >= DIGIT_REPEAT_S:
                last_digit[key] = now
                commands.put(("rays", 0))
            return
        if key in (ord(MOUNT_SWITCH_KEY), ord(MOUNT_SWITCH_KEY.upper())):
            now = time.monotonic()
            if now - last_digit.get(ord(MOUNT_SWITCH_KEY), -1.0) >= MOUNT_SWITCH_REPEAT_S:
                last_digit[ord(MOUNT_SWITCH_KEY)] = now
                commands.put(("mount", 0))
            return
        keyboard.feed(key, source=source)

    def save_scene(env: Any, task_scene: PlannerScene) -> None:
        """Compile the current model for the worker: one temp file, the previous one removed."""
        import mujoco

        nonlocal scene_version, scene_init, scene_path
        scene_version += 1
        SCENE_DIR.mkdir(parents=True, exist_ok=True)
        path = SCENE_DIR / "scene_{}_{}.mjb".format(os.getpid(), scene_version)
        mujoco.mj_saveModel(env.sim.model._model, str(path), None)
        _remove_quietly(scene_path)
        scene_path = path
        scene_init = {"model_path": str(path), "arm_qpos_adr": task_scene.arm_qpos_adr,
                      "arm_dof_adr": task_scene.arm_dof_adr, "arm_joint_ids": task_scene.arm_joint_ids,
                      "eef_site": task_scene.eef_site, "robot_geoms": sorted(task_scene.robot_geoms)}

    def build_task(env: Any, hand_spec: LiberoTeleopSpec,
                   rays_on: bool) -> tuple[PlannerActionComposer, PlannerScene, FingertipRays]:
        """Composer, targets and rays for the current model (start, R, M)."""
        task_composer = PlannerActionComposer(env, hand_spec)
        task_composer.initialize_goal()
        task_scene = PlannerScene(env, task_composer)
        task_rays = FingertipRays(env, task_scene.robot_geoms, enabled=rays_on)
        if planner_backend == "ompl":
            save_scene(env, task_scene)
        return task_composer, task_scene, task_rays

    def request_target(number: int) -> None:
        assert scene is not None and composer is not None
        target = scene.by_number(number)
        if target is None:
            print("no target {} in this task (1-{})".format(number, len(scene.targets)))
            return
        held = scene.held()
        if any(t.number == target.number for t in held):
            print("target {} {} is in the hand".format(target.number, target.name))
            return
        hand = scene.hand_state(held)
        box = scene.box(target)
        goal_pos = approach_goal(hand, box, clearance_m)
        goal_rot = hand.ee_rot.copy()
        obstacles = [scene.box(t) for t in scene.targets if t not in held]
        if planner_backend == "straight":
            payload: dict[str, Any] = {"waypoints": straight_waypoints(hand, goal_pos, obstacles)}
        else:
            data = scene.env.sim.data._data
            payload = {
                "scene": {"version": scene_version, "init": dict(scene_init)},
                "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
                "goal_pos": goal_pos.tolist(), "goal_rot": goal_rot.ravel().tolist(),
                "held": [{"body": t.root_body, "geoms": list(t.geoms)} for t in held],
                "candidates": [[np.asarray(p, dtype=np.float64).tolist() for p in route]
                               for route in cartesian_candidates(hand, goal_pos, obstacles)],
                "time_limit_s": OMPL_TIME_LIMIT_S, "simplify_s": OMPL_SIMPLIFY_S, "seed": planner_counts["requests"],
            }
        planner_counts["requests"] += 1
        planner.start(target, goal_pos, goal_rot, hand, [t.name for t in held], payload)
        print("planning ({}) -> {} {}{}".format(planner_backend, target.number, target.name,
                                                " carrying " + ", ".join(t.name for t in held) if held else ""))

    def rebind(env: Any) -> None:
        """After R or M: inputs back to neutral, views on the (new) model."""
        mapper.reset()
        device.reset_input_state()
        hand_input.reset()
        hand_resampler.reset(discard_until_s=time.perf_counter())
        gestures.reset(buttons=last_buttons)
        tri_view.track_site = composer.controller.eef_name
        tri_view.configure_cameras(env)
        mapper.set_camera_from_sim(env, tri_config.front_camera_name)

    planner = ApproachPlanner(_straight_plan, "straight", max_speed_m_s=planner_speed_m_s)
    timer_resolution = _WindowsTimerResolution().__enter__()
    try:
        _remove_stale_scenes(time.time())
        if planner_backend == "ompl":
            python = discover_planner_python(planner_python)
            if python is None:
                raise RuntimeError("OMPL planner Python not found; set OMPL_PLANNER_PYTHON or --planner-python "
                                   "(or use --planner straight)")
            sidecar = OmplPlannerSidecar(python)
            planner = ApproachPlanner(_OmplBridge(sidecar).plan, "ompl", max_speed_m_s=planner_speed_m_s)
        device_info = device.open()
        session.open()
        env = session.env
        assert env is not None
        substeps = int(env.control_timestep / env.model_timestep)
        dt_sim = substeps * env.model_timestep
        composer, scene, rays = build_task(env, spec, rays_enabled)
        if session.object_mass_scale != 1.0 or session.object_mass_overrides_kg:
            print(describe_object_masses(session))
        tri_config = follow_robot_base(_shadow_tri_view_config(config), env)
        embed_preview = bool(tri_config.enabled and not headless)

        def extras() -> list[ExtraGeom]:
            return rays.geoms() if rays is not None else []

        def badge_points() -> list[tuple[str, np.ndarray]]:
            return scene.badge_points() if scene is not None else []

        tri_view = _planner_view_window(
            tri_config, env, composer, headless=headless, view_size=view_size,
            key_handler=lambda key: route(key, "views"),
            on_close=lambda: keyboard.feed("q", source="views_window"),
            preview=(lambda: camera.latest_frame()[0] if camera is not None and camera_display else None),
            scene_extras=extras, badge_points=badge_points,
        )
        mapper.set_camera_from_sim(env, tri_config.front_camera_name)
        if hand_control == "dex":
            print("Starting official dex-retargeting sidecar for Shadow (first launch can take ~60 s)...")
            retargeter = (ShadowDexRetargeter(thumb_lift_gain=thumb_lift_gain) if dex_grasp_terms
                          else ShadowDexRetargeter(grasp_terms_path=None))
            hand_input = ShadowDexHandInput(
                retargeter, composer.gripper.lower_limits, composer.gripper.upper_limits,
                stale_timeout_s=config.dex_hand.stale_timeout_s,
                expected_handedness=config.dex_hand.expected_handedness, unmirror_world_x=camera_mirror,
            )
        else:
            hand_input = ShadowLegacyHandInput(composer.gripper.lower_limits, composer.gripper.upper_limits)
        gestures = ShadowHandGestureController(
            config=config.hand_gestures,
            lower_limits=composer.gripper.lower_limits, upper_limits=composer.gripper.upper_limits,
        )
        gestures.reset(buttons=last_buttons)
        hand_resampler = FixedDelayResampler(len(composer.gripper.lower_limits), config.hand_target_delay_s)
        if hand_control in ("dex", "legacy-curl"):
            camera = CameraHandWorker(
                hand_input, camera_index=camera_index, display=camera_display and not headless,
                mirror_input=camera_mirror, keypress_callback=lambda key: route(key, "camera"),
            )
            camera.start()
        log.metadata({
            "mode": "libero_planner_teleoperation", "spec": spec.name, "hand_mount": mount,
            "mount_switch_key": MOUNT_SWITCH_KEY, "session": session.describe(),
            "device": device_info, "hand_control": hand_control, "camera_mirror": camera_mirror,
            "hand_target_delay_ms": 1e3 * config.hand_target_delay_s,
            "dex": retargeter.metadata() if retargeter is not None else None,
            "sim_speed": {"requested": sim_speed, "control_freq_hz": 1.0 / env.control_timestep,
                          "substeps": substeps, "sim_s_per_tick": dt_sim,
                          "effective": dt_sim * config.control_hz, "wall_hz": config.control_hz},
            "planner": {"backend": planner_backend, "clearance_m": clearance_m,
                        "max_speed_m_s": planner.max_speed_m_s,
                        "max_angular_speed_rad_s": planner.max_angular_speed_rad_s,
                        "ompl_time_limit_s": OMPL_TIME_LIMIT_S, "scene_dir": str(SCENE_DIR),
                        "python": str(discover_planner_python(planner_python)) if planner_backend == "ompl" else None,
                        "targets": [{"number": t.number, "name": t.name, "kind": t.kind} for t in scene.targets]},
            "arm": {
                "translation_speed_scale": arm_speed_scale, "rotation_speed_scale": rotation_speed_scale,
                "translation_scale_m": config.translation_scale_m, "max_translation_delta_m": config.max_translation_delta_m,
                "rotation_scale_rad": config.rotation_scale_rad, "max_rotation_delta_rad": config.max_rotation_delta_rad,
                "goal_leash_m": LIBERO_GOAL_LEASH_M, "goal_leash_rad": LIBERO_GOAL_LEASH_RAD, "osc_kp": session.osc_kp,
            },
            "rays": {"enabled": rays.enabled, "fingers": list(FINGERS), "direction": "world -z"},
            "views": tri_view.camera_setup, "control_frame": mapper.describe_control_frame(),
        })
        print("LIBERO {} task {} ready: {} | {}".format(session.suite, session.task_id, spec.name, session.task.language))
        print("hand mount: {} ({}) | sim speed x{:g} ({:.0f} ms simulated per 50 ms tick) | planner: {}, "
              "up to {:.2f} m/s".format(mount, MOUNT_TEXT[mount], dt_sim * config.control_hz, 1e3 * dt_sim,
                                        planner_backend, planner.max_speed_m_s))
        print("targets:\n" + scene.table())
        print("Keys: 1-9 approach a target | 0 cancel | L fingertip rays | V close-up | M switch hand mount "
              "(restarts the task) | R reset | Q quit. Moving the SpaceMouse stops the planner and takes over.")
        started = previous = clock.restart()
        smoke_pending = smoke_target
        smoke_queue = sorted((float(at), str(key)) for at, key in smoke_keys)
        while duration_s is None or time.perf_counter() - started < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                reset_feedback.begin(tri_view)
                planner.cancel("reset")
                session.reset()
                composer, scene, rays = build_task(env, spec, rays.enabled)
                rebind(env)
                log.event({"event": "environment_reset", "hand_mount": mount, "timestamp_monotonic_s": time.monotonic()})
                previous = clock.restart()
                continue
            if smoke_pending is not None and time.perf_counter() - started > 0.5:
                commands.put(("target", int(smoke_pending)))
                smoke_pending = None
            while smoke_queue and time.perf_counter() - started >= smoke_queue[0][0]:
                route(ord(smoke_queue.pop(0)[1][0]), "smoke")
            switch_requested = False
            while True:
                try:
                    command, value = commands.get_nowait()
                except queue.Empty:
                    break
                if command == "target":
                    request_target(value)
                elif command == "cancel":
                    if planner.cancel("key 0"):
                        composer.hold_here()
                        print("planner cancelled (0)")
                elif command == "rays":
                    rays.enabled = not rays.enabled
                    print("fingertip rays {}".format("on" if rays.enabled else "off"))
                elif command == "mount":
                    switch_requested = True
            if switch_requested:
                # M: the same task and state again, with the other hand mount.
                reset_feedback.begin(tri_view, label="hand mount switch", notice=MOUNT_SWITCH_NOTICE)
                planner.cancel("hand mount switch")
                new_mount = other_mount(mount)
                print("switching hand mount: {} -> {} ({}); rebuilding the task ...".format(
                    mount, new_mount, MOUNT_TEXT[new_mount]))
                switch_started = time.perf_counter()
                session.close()
                spec = shadow_spec(new_mount)
                session = make_session(spec)
                session.open()
                env = session.env
                assert env is not None
                mount = new_mount
                composer, scene, rays = build_task(env, spec, rays.enabled)
                rebind(env)
                while True:  # keys pressed during the rebuild are stale
                    try:
                        commands.get_nowait()
                    except queue.Empty:
                        break
                planner_counts["mount_switches"] += 1
                rebuild_s = time.perf_counter() - switch_started
                log.event({"event": "hand_mount_switch", "hand_mount": mount, "spec": spec.name,
                           "rebuild_s": rebuild_s, "session": session.describe(),
                           "targets": [{"number": t.number, "name": t.name, "kind": t.kind} for t in scene.targets],
                           "timestamp_monotonic_s": time.monotonic()})
                print("hand mount now {} ({}), task restarted in {:.1f} s".format(mount, MOUNT_TEXT[mount], rebuild_s))
                previous = clock.restart()
                continue
            loop_start = time.perf_counter()
            tick = time.monotonic()
            sample = device.poll()
            last_buttons = sample.buttons if sample.buttons_valid else (False,) * len(sample.buttons)
            wrist = mapper.map(sample.control_axes, valid=bool(sample.valid and sample.armed))
            period = loop_start - previous
            wrist = wrist * float(np.clip(period * config.control_hz, 1.0, 2.0))
            camera_command = hand_input.latest(tick)
            if camera is not None:
                hand_sample_seq, new_samples = camera.samples_since(hand_sample_seq)
                for frame_s, frame_targets in new_samples:
                    hand_resampler.push(frame_s, frame_targets)
            camera_targets = camera_command.joint_targets
            if camera_command.valid:
                delayed_targets = hand_resampler.sample(loop_start)
                if delayed_targets is not None:
                    camera_targets = delayed_targets
            hand_command = gestures.update(
                last_buttons, camera_targets,
                camera_valid=bool(camera_command.valid and hand_control in ("dex", "legacy-curl")), timestamp_s=tick,
            )
            hand_value, hand_state = hand_command.joint_targets, hand_command.source
            # One owner per tick: SpaceMouse input takes the arm back at once.
            if planner.active and np.any(wrist != 0.0):
                planner.cancel("spacemouse")
                composer.hold_here()
                print("SpaceMouse took over")
            ee_pos, ee_rot = composer.ee_pose()
            goal = planner.tick(dt_sim, ee_pos, ee_rot)
            if goal is not None:
                composer.set_goal(*goal)
                action = composer.compose(np.zeros(6), hand_value)
                owner = "planner"
            else:
                action = composer.compose(wrist, hand_value)
                owner = "spacemouse"
            for event in planner.drain_events():
                event["timestamp_monotonic_s"] = tick
                log.event(event)
                kind = event["event"]
                if kind == "planner_result":
                    print(_describe_plan(event))
                elif kind == "planner_arrived":
                    planner_counts["arrived"] += 1
                    print("arrived (end effector {:.1f} mm from goal); SpaceMouse has the arm".format(
                        1e3 * event["ee_error_m"]))
                elif kind == "planner_cancel":
                    planner_counts["cancelled"] += 1
                elif kind == "planner_failed":
                    planner_counts["failed"] += 1
                    print("planning failed{}: {}".format(
                        " ({})".format(event["reason"]) if event.get("reason") else "", event["error"]))
            step_started = time.perf_counter()
            _obs, reward, _done, _info = env.step(action)
            step_ms = 1e3 * (time.perf_counter() - step_started)
            visual = tri_view.update(env, status_text="LIBERO {} {} | mount {} | sim x{:g} | {} | hand={} | rays {}".format(
                session.suite, session.task_id, mount, dt_sim * config.control_hz, planner.status(), hand_state,
                "on" if rays.enabled else "off"))
            if not screenshot_written and tri_view.last_frame is not None:
                screenshot = log.path / "tri_view_initial.png"
                visual["screenshot"] = str(screenshot)
                screenshot_written = bool(tri_view.save_last_frame(screenshot))
            if visual.get("closed"):
                keyboard.feed("q", source="views")
            loop_periods.append(max(period, 1e-9))
            previous = loop_start
            log.event({
                "event": "step", "timestamp_monotonic_s": tick,
                "timing_ms": {"step": step_ms, "busy": 1e3 * (time.perf_counter() - loop_start)},
                "input_valid": sample.valid, "input_armed": sample.armed, "input_error": sample.error,
                "buttons": sample.buttons, "arm_owner": owner, "planner_state": planner.state, "hand_mount": mount,
                "arm_action": action[composer.arm_slice], "hand_action": action[composer.hand_slice],
                "mapped_world_delta": wrist, "ee_pos": ee_pos,
                "hand": {"source": hand_state, "camera_valid": camera_command.valid,
                         "camera_tracking_state": camera_command.tracking_state,
                         "camera_error": camera_command.error, "resampling": hand_resampler.describe()},
                "reward": reward, "task_success": bool(env._check_success()), "sim_time_s": float(env.sim.data.time),
            })
            if reset_feedback.active:  # first step on the new model done: the scene is back
                log.event(reset_feedback.settle(tri_view))
            clock.wait()
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        exit_code, reason = 1, "error: {}: {}".format(type(exc).__name__, exc)
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
    finally:
        display: dict[str, Any] = {}
        loop_hz = None
        try:
            planner.cancel("exit")
            if composer is not None and session.env is not None:
                composer.hold_here()
                final = composer.compose(np.zeros(6), None)
                session.env.step(final)
                log.event({"event": "safe_zero", "action": final, "timestamp_monotonic_s": time.monotonic()})
            if tri_view is not None:
                display = tri_view.display_stats()
                tri_view.close()
            if camera is not None:
                camera.close()
            if retargeter is not None:
                retargeter.close()
            if sidecar is not None:
                sidecar.close()
            mapper.reset()
            device.close()
            session.close()
        finally:
            _remove_quietly(scene_path)
            timer_resolution.__exit__()
            loop_hz = 1.0 / float(np.median(loop_periods[1:])) if len(loop_periods) > 1 else None
            log.event({"event": "exit", "reason": reason, "median_loop_hz": loop_hz, "display": display,
                       "planner": dict(planner_counts),
                       "clock": {"period_ms": 1e3 * clock.period_s, "ticks": clock.ticks,
                                 "overruns_dropped_ticks": clock.overruns},
                       "timestamp_monotonic_s": time.monotonic()})
            log.close()
        if loop_hz is not None:
            print("median control loop: {:.1f} Hz (target {:.0f} Hz)".format(loop_hz, config.control_hz))
        print("LIBERO planner teleoperation stopped: {}. Log: {}".format(reason, log.path))
    return exit_code
