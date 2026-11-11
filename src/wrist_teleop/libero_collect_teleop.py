"""LIBERO data collection entry: the planner entry plus a live success check, a CSV and videos recorded after S.

Copy of the planner entry's main loop (``libero_planner_teleop``, main
command 2, unchanged): SpaceMouse -> arm, camera -> Shadow fingers; number
keys 1-9 send the hand above a task object / fixture with the approach
planner (moving the SpaceMouse stops it and takes the arm back at once, ``0``
cancels); five red fingertip rays (``L`` toggles); number badges in the
views; ``M`` switches the hand mount. Added for data collection:

* LIBERO's own random placement (BDDL regions), seeded per placement, instead
  of the official evaluation init states; the dropped objects settle first;
* a success monitor generated from each task's BDDL goal: every condition is
  shown live, the decision is the official ``env._check_success()`` held
  ``SUCCESS_HOLD_STEPS`` steps;
* every attempt the operator starts gets one row in ``attempts.csv``
  (success / failed / quit, the time, ...); after a success the next
  placement loads, and the next task once a task has its successes;
* a video only after the operator presses ``S``: from that moment (from 1 s
  before the first move when pressed before it) to the end of the attempt,
  the window exactly as shown (webcam inset included) with the timer and the
  result on top (``libero_attempt_video``). It is written in the run log
  folder and moved into ``videos/`` when the attempt ends; S again cancels
  it (ffmpeg stopped, the file deleted). Without S nothing is recorded;
* the close-up is the planner entry's close-up turned twice with V (camera on
  the robot's left, looking at the hand), fixed there; only its image is
  mirrored left-right.

The collection folder gets that table and the videos only. The raw
simulator demos (states / actions, for training later) are written only
with ``save_demos`` (``--save-demos``), into a separate folder.

Real time only (sim speed 1): LIBERO's 20 Hz control steps. One arm owner per
tick, as in the planner entry: the SpaceMouse, or the planner after a number
key (a raw demo records which at every step, ``control/arm_owner``). The
fingers ``[6:26]`` always come from the camera hand.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .config import WristConfig
from .libero_collect_device import KeepNeutralSpaceMouse
from .hand_input import CameraHandWorker
from .libero_attempt_video import (
    FINISH_TIMEOUT_S,
    PREROLL_S,
    RESULT_HOLD_S,
    VIDEO_CRF,
    VIDEO_PRESET,
    VIDEO_WIDTH,
    AttemptVideo,
)
from .libero_collection import (
    ARM_OWNERS,
    ATTEMPTS_CSV,
    COLLECT_ROOT,
    DEMO_FORMAT,
    DEMO_ROOT,
    DEMOS_PER_TASK,
    DISTURB_MOVE_M,
    DISTURB_TILT_DEG,
    RUN_LOG_ROOT,
    SUCCESS_HOLD_STEPS,
    VIDEO_DIR,
    AttemptTable,
    DemoRecorder,
    DemoStore,
    DisturbanceMonitor,
    TaskSuccessMonitor,
    attempt_outcome,
    attempt_video_name,
    collection_panel,
    describe_disturbance,
    ended_by_text,
    format_summary,
    goal_object_names,
    placement_seed,
)
from .libero_motion_planning import (
    FINGERS,
    PLANNER_BACKENDS,
    PLANNER_MAX_SPEED_M_S,
    PLANNER_SPEED_RANGE_M_S,
    PREGRASP_CLEARANCE_M,
    ApproachPlanner,
    OmplPlannerSidecar,
    approach_goal,
    cartesian_candidates,
    discover_planner_python,
    straight_waypoints,
)
from .libero_planner_teleop import (
    DIGIT_REPEAT_S,
    MOUNT_SWITCH_KEY,
    MOUNT_SWITCH_NOTICE,
    MOUNT_SWITCH_REPEAT_S,
    MOUNT_TEXT,
    OMPL_SIMPLIFY_S,
    OMPL_TIME_LIMIT_S,
    PLANNER_SUITES,
    SCENE_DIR,
    FingertipRays,
    PlannerActionComposer,
    PlannerScene,
    PlannerViewWindow,
    _describe_plan,
    _OmplBridge,
    _remove_quietly,
    _remove_stale_scenes,
    _straight_plan,
    arm_config_for_speeds,
    other_mount,
)
from .libero_spatial_teleop import (
    DETAIL_FOLLOW,
    LIBERO_ARM_ROTATION_SPEED_SCALE,
    LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    LIBERO_GOAL_LEASH_M,
    LIBERO_GOAL_LEASH_RAD,
    SCENE_FOLLOW,
    LiberoTeleopSpec,
    OfficialLiberoSpatialSession,
    _NeutralInput,
    _WindowsTimerResolution,
    _json_default,
    _shadow_tri_view_config,
    configure_local_libero_paths,
    describe_object_masses,
    ensure_windows_libero_runtime,
    follow_robot_base,
    scale_object_masses,
    shadow_spec,
)
from .libero_views import DETAIL_ORBIT_STEP_DEG, Panel, _text, default_view_size, enable_dpi_awareness
from .loop_timing import FixedDelayResampler, FixedRateClock
from .mapper import SixDofMapper
from .runtime_controls import KeyboardCommandRouter, ResetFeedback
from .shadow_dex import ShadowDexHandInput, ShadowDexRetargeter
from .shadow_gripper import SHADOW_MOUNTS, register_shadow_gripper
from .shadow_hand_input import ShadowHandGestureController, ShadowLegacyHandInput

# The planner entry's close-up after V twice: 90 + 2 x 90 = 270 deg, the
# camera on the robot's left looking at the hand. Fixed there (V does nothing
# here) and shown mirrored left-right; distance, height, zoom and follow are
# the planner entry's.
COLLECT_DETAIL_FOLLOW = replace(DETAIL_FOLLOW, azimuth_deg=DETAIL_FOLLOW.azimuth_deg + 2.0 * DETAIL_ORBIT_STEP_DEG,
                                orbit=False)
CLOSE_UP_LABEL = "CLOSE-UP (follows hand)"  # the planner entry's label, unchanged
SIDE_LABEL = "3D FROM FRONT-RIGHT (follows hand)"
SIM_SPEED = 1.0  # real time: the demos are LIBERO's 20 Hz control steps
ALL_DONE_SHOW_S = 1.5
RECORD_KEY = "s"  # starts recording this attempt (kept when it ends); again: cancels it. Nothing recorded without it
RECORD_REPEAT_S = 0.3  # presses closer than this to the previous one (a held key) do not toggle again
PLACEMENT = "libero_random"
# A random placement drops the objects from a few cm; they can slide 5-10 cm
# in the first 0.3-1 s. Before each attempt the arm holds still until every
# object is slower than SETTLE_SPEED_MPS for SETTLE_CALM_STEPS steps in a row
# (at most SETTLE_MAX_STEPS steps), like the wait steps after a reset in
# LIBERO's own evaluation. These steps are not recorded.
SETTLE_MAX_STEPS = 60
SETTLE_SPEED_MPS = 0.005
SETTLE_CALM_STEPS = 3
# Planner event fields kept with each demo (the run telemetry keeps them all).
PLANNER_NOTE_FIELDS = {
    "planner_request": ("number", "target", "kind", "held"),
    "planner_result": ("strategy", "path_length_m", "duration_sim_s", "plan_ms"),
    "planner_arrived": ("within_tolerance", "ee_error_m"),
    "planner_cancel": ("reason", "state"),
    "planner_failed": ("error", "reason"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- session
class CollectionSession(OfficialLiberoSpatialSession):
    """One official LIBERO task with LIBERO's own random placement, seeded per placement.

    Same environment, controller and hand as the parent (whose official
    evaluation init states are left for evaluation); no reference env.
    ``env.reset()`` samples the BDDL placement regions with ``np.random``,
    so seeding it first makes each placement reproducible.
    """

    def __init__(self, spec: LiberoTeleopSpec, config: WristConfig, *, suite: str, task_id: int, seed: int,
                 object_mass_scale: float = 1.0, object_masses_kg: Optional[dict[str, float]] = None) -> None:
        super().__init__(spec, config, task_id=task_id, task_order=0, init_state_id=0, seed=seed,
                         has_offscreen_renderer=True, suite=suite, object_mass_scale=object_mass_scale,
                         object_masses_kg=object_masses_kg)
        self.placement_seed: Optional[int] = None
        self.bddl_path: Optional[Path] = None
        self.problem_info: dict[str, Any] = {}
        self.env_kwargs: dict[str, Any] = {}

    def open(self) -> None:
        configure_local_libero_paths()
        ensure_windows_libero_runtime()
        if self.spec.camera_hand:
            register_shadow_gripper()
        try:
            from libero.libero import get_libero_path
            from libero.libero.benchmark import get_benchmark
            from libero.libero.envs import bddl_utils
            from libero.libero.envs.bddl_base_domain import TASK_MAPPING
        except ImportError as exc:
            raise RuntimeError("official LIBERO dependencies are unavailable in .venv-libero") from exc
        benchmark = get_benchmark(self.suite)(self.task_order)
        if self.task_id >= benchmark.get_num_tasks():
            raise ValueError("task-id {} is outside official {} range (0-{})".format(
                self.task_id, self.suite, benchmark.get_num_tasks() - 1))
        self.task = benchmark.get_task(self.task_id)
        self.bddl_path = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        common = dict(
            bddl_file_name=str(self.bddl_path), robots=["Panda"], use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=True,
            control_freq=int(round(self.config.control_hz)), horizon=10_000_000,
            ignore_done=True, camera_names=list(self.config.tri_view.camera_names),
            camera_heights=128, camera_widths=128, initialization_noise=None,
        )
        self.problem_info = dict(bddl_utils.get_problem_info(str(self.bddl_path)))
        controller = self._controller()
        self.env = TASK_MAPPING[self.problem_info["problem_name"]](
            **common, controller_configs=controller, gripper_types=self.spec.gripper_type,
        )
        self.env_kwargs = dict(common, controller_configs=controller, gripper_types=self.spec.gripper_type)
        self.env.seed(self.seed)

    def reset(self) -> Any:
        if self.placement_seed is None:
            raise RuntimeError("use reset_to(seed) in the collection entry")
        return self.reset_to(self.placement_seed)

    def reset_to(self, seed: int) -> Any:
        """A new LIBERO random placement drawn with ``seed`` (robosuite 1.4 rebuilds the model)."""

        if self.env is None:
            raise RuntimeError("open the LIBERO session before reset")
        np.random.seed(int(seed) % (2 ** 32))
        self.env.reset()
        self.env.sim.forward()
        self.object_masses = scale_object_masses(self.env, self.object_mass_scale, self.object_mass_overrides_kg)
        self.env._post_process()
        self.env._update_observables(force=True)
        self.placement_seed = int(seed)
        return self.env._get_observations()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        for key in ("init_state_id", "copied_joint_count", "copied_joint_names"):
            info.pop(key, None)
        info.update({
            "placement": PLACEMENT, "placement_seed": self.placement_seed,
            "initial_state_strategy": "libero_random_placement_seeded_numpy",
            "problem_info": self.problem_info, "bddl_path": str(self.bddl_path),
        })
        return info


# --------------------------------------------------------------- views
def _draw_panel(cv2: Any, frame: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int]]],
                rect: tuple[int, int, int, int], ui_scale: float) -> None:
    """Right-aligned lines in the lower right of the main view (the webcam inset is lower left)."""

    scale = 0.6 * ui_scale
    pad = int(round(14 * ui_scale))
    step = int(round(30 * ui_scale))
    thickness = max(1, int(round(scale * 1.6)))
    x0, y0, width, height = rect
    y = y0 + height - pad - step * (len(lines) - 1)
    for text, color in lines:
        (text_w, _text_h), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        _text(cv2, frame, text, (max(x0 + pad, x0 + width - pad - text_w), y), scale, color)
        y += step


class CollectionViewWindow(PlannerViewWindow):
    """The planner entry's views (fingertip rays, target badges) with the close-up mirrored and the collection panel.

    Only the close-up's image is flipped left-right; its camera, rays and
    follow are the planner entry's. Labels are drawn after the flip, so they
    read normally. ``panel_lines`` is set by the control loop before each
    update (the render thread draws the latest lines).
    """

    MIRRORED_SLOTS = ("wrist",)
    panel_lines: tuple[tuple[str, tuple[int, int, int]], ...] = ()

    def _decorate(self, cv2: Any, frame: np.ndarray, status_text: Optional[str]) -> None:
        for slot in self.MIRRORED_SLOTS:
            x, y, width, height = self.rects[slot]
            frame[y:y + height, x:x + width] = frame[y:y + height, x:x + width][:, ::-1].copy()
        super()._decorate(cv2, frame, status_text)
        lines = self.panel_lines
        if lines:
            _draw_panel(cv2, frame, lines, self.rects["front"], self.ui_scale)


def collection_view_window(
    tri_config: Any, env: Any, composer: PlannerActionComposer, *, headless: bool,
    view_size: Optional[tuple[int, int]], key_handler: Any, on_close: Any, preview: Any,
    scene_extras: Any, badge_points: Any,
) -> CollectionViewWindow:
    """The planner entry's three panels; the close-up is its V x2 view, fixed and mirrored."""

    panels = (
        Panel("front", "", camera=tri_config.front_camera_name),
        Panel("wrist", CLOSE_UP_LABEL, follow=COLLECT_DETAIL_FOLLOW),
        Panel("side", SIDE_LABEL, follow=SCENE_FOLLOW),
    )
    window = CollectionViewWindow(
        tri_config, panels, size=view_size or default_view_size(), show_window=not headless,
        track_site=composer.controller.eef_name, key_handler=key_handler, on_close=on_close,
        laser=None, preview=preview, scene_extras=scene_extras, badge_points=badge_points,
    )
    window.configure_cameras(env)
    return window


# ------------------------------------------------------------- helpers
def object_poses(env: Any, names: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = env.sim.data
    poses = {}
    for name in names:
        body = env.obj_body_id.get(name)
        if body is not None:
            poses[name] = (np.array(data.body_xpos[body], dtype=np.float64),
                           np.array(data.body_xmat[body], dtype=np.float64).reshape(3, 3))
    return poses


def flat_state(env: Any) -> np.ndarray:
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)


def settle_scene(env: Any, composer: Any, *, max_steps: int = SETTLE_MAX_STEPS,
                 speed_mps: float = SETTLE_SPEED_MPS, calm_steps: int = SETTLE_CALM_STEPS) -> dict[str, Any]:
    """Step with a zero arm command (hand open) until the dropped objects have landed."""

    names = sorted(env.objects_dict)
    dt = float(env.control_timestep)
    previous = object_poses(env, names)
    steps = calm = 0
    fastest = 0.0
    while steps < max_steps and calm < calm_steps:
        env.step(composer.compose(np.zeros(6), None))
        steps += 1
        current = object_poses(env, names)
        fastest = max((float(np.linalg.norm(current[name][0] - previous[name][0])) / dt for name in current),
                      default=0.0)
        calm = calm + 1 if fastest < speed_mps else 0
        previous = current
    return {"steps": steps, "sim_s": round(steps * dt, 3), "max_speed_mps": round(fastest, 4),
            "settled": calm >= calm_steps}


def operator_camera(env: Any, name: str) -> dict[str, Any]:
    """The main-view camera as the operator saw it (the view window moves it at startup)."""

    import mujoco

    model = env.sim.model._model
    camera = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name))
    return {"name": name, "pos": model.cam_pos[camera].tolist(), "quat_wxyz": model.cam_quat[camera].tolist(),
            "fovy_deg": float(model.cam_fovy[camera])}


def software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("robosuite", "mujoco", "h5py"):
        try:
            versions[name] = str(__import__(name).__version__)
        except Exception:
            versions[name] = "unavailable"
    return versions


def suite_task_table(suite: str) -> list[dict[str, Any]]:
    """Every task of a suite with its BDDL goal, without building an environment."""

    configure_local_libero_paths()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import bddl_utils

    benchmark = get_benchmark(suite)(0)
    rows = []
    for task_id in range(benchmark.get_num_tasks()):
        task = benchmark.get_task(task_id)
        path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        rows.append({"task_id": task_id, "task_name": task.name, "language": task.language,
                     "goal": [list(state) for state in bddl_utils.robosuite_parse_problem(str(path))["goal_state"]]})
    return rows


def planner_note(event: dict[str, Any], step: int) -> dict[str, Any]:
    """The part of a planner event kept with the demo: what happened, at which recorded step, a few fields."""

    kind = str(event.get("event", ""))
    note: dict[str, Any] = {"step": int(step), "event": kind.replace("planner_", "", 1)}
    for key in PLANNER_NOTE_FIELDS.get(kind, ()):
        if key in event:
            value = event[key]
            note[key] = round(float(value), 4) if isinstance(value, (float, np.floating)) else value
    return note


class _CollectLog:
    """``<log root>/<stamp>_collect_<suite>/`` (outputs/libero/ by default, like the other entries' logs).

    metadata.json and a compact telemetry.jsonl, for debugging; the
    collection folder itself holds only attempts.csv and the videos.
    """

    def __init__(self, log_root: Path, suite: str) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(log_root) / "{}_collect_{}".format(self.run_id, suite)
        path, suffix = base, 1
        while True:
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                path = base.with_name("{}_{}".format(base.name, suffix))
        self.path = path
        self._stream = (self.path / "telemetry.jsonl").open("w", encoding="utf-8")

    def metadata(self, contents: dict[str, Any]) -> None:
        (self.path / "metadata.json").write_text(json.dumps(contents, indent=2, default=_json_default), encoding="utf-8")

    def event(self, contents: Optional[dict[str, Any]]) -> None:
        if contents:
            self._stream.write(json.dumps(contents, default=_json_default, separators=(",", ":")) + "\n")
            self._stream.flush()

    def close(self) -> None:
        self._stream.close()


@dataclass
class CollectResult:
    exit_code: int
    reason: str
    run_dir: Optional[Path]  # the run log
    suite_root: Path  # attempts.csv and videos/
    table_path: Path
    successes: int = 0  # this run
    videos: list[Path] = field(default_factory=list)  # this run's finished videos
    demos: list[Path] = field(default_factory=list)  # raw demos (save_demos only)
    tasks_touched: list[int] = field(default_factory=list)


# ----------------------------------------------------------- main loop
def run_libero_collect(
    config: WristConfig,
    *,
    suite: str,
    tasks: Sequence[int],
    demos_per_task: int = DEMOS_PER_TASK,
    hand_mount: str = "down",
    seed: int = 7,
    headless: bool = False,
    duration_s: Optional[float] = None,
    camera_index: int = 0,
    camera_display: bool = True,
    camera_mirror: bool = True,
    hand_control: str = "dex",
    synthetic_input: bool = False,
    arm_speed_scale: float = LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    rotation_speed_scale: float = LIBERO_ARM_ROTATION_SPEED_SCALE,
    object_mass_scale: float = 1.0,
    object_masses_kg: Optional[dict[str, float]] = None,
    view_size: Optional[tuple[int, int]] = None,
    dex_grasp_terms: bool = True,
    thumb_lift_gain: Optional[float] = None,
    planner_backend: str = "ompl",
    planner_python: Optional[str] = None,
    clearance_m: float = PREGRASP_CLEARANCE_M,
    planner_speed_m_s: float = PLANNER_MAX_SPEED_M_S,
    rays_enabled: bool = True,
    output_root: Path = COLLECT_ROOT,
    operator: str = "",
    hold_steps: int = SUCCESS_HOLD_STEPS,
    save_demos: bool = False,
    demo_root: Path = DEMO_ROOT,
    log_root: Path = RUN_LOG_ROOT,
    device: Any = None,
    step_hook: Optional[Callable[[Any, Any], Optional[str]]] = None,
) -> CollectResult:
    """Collect ``demos_per_task`` successes for each of ``tasks`` (skipping full ones).

    Writes ``<output_root>/<suite>/attempts.csv`` and, for each started
    attempt where ``S`` was pressed, its video from that press in
    ``videos/``; raw demos into ``<demo_root>/<suite>/`` only with
    ``save_demos``; the run log into ``log_root``. ``device`` replaces the
    SpaceMouse and ``step_hook(env, info)`` runs before every step and may
    return a key to press (tests / headless smoke only; ``info`` has
    ``attempt``, ``step``, ``task_id``, ``mount`` and ``planner``, the
    planner's state).
    """

    if suite not in PLANNER_SUITES:
        raise ValueError("the collection entry covers {} only".format(" and ".join(PLANNER_SUITES)))
    if hand_mount not in SHADOW_MOUNTS:
        raise ValueError("hand mount must be one of {}".format(", ".join(SHADOW_MOUNTS)))
    if hand_control not in ("dex", "legacy-curl", "off"):
        raise ValueError("hand control must be dex, legacy-curl, or off")
    if int(demos_per_task) < 1:
        raise ValueError("demos per task must be at least 1")
    if planner_backend not in PLANNER_BACKENDS:
        raise ValueError("planner must be one of {}".format(", ".join(PLANNER_BACKENDS)))
    if not np.isfinite(clearance_m) or not 0.0 <= clearance_m <= 0.2:
        raise ValueError("clearance must be within 0-0.2 m")
    low_speed, high_speed = PLANNER_SPEED_RANGE_M_S
    if not np.isfinite(planner_speed_m_s) or not low_speed <= planner_speed_m_s <= high_speed:
        raise ValueError("planner speed must be within {:g}-{:g} m/s".format(low_speed, high_speed))
    env_config = config  # the LIBERO env runs at config.control_hz (sim speed 1)
    config = arm_config_for_speeds(config, arm_speed_scale, rotation_speed_scale, SIM_SPEED)
    demos_per_task = int(demos_per_task)
    task_rows = suite_task_table(suite)
    task_names = {row["task_id"]: row["task_name"] for row in task_rows}
    languages = {row["task_id"]: row["language"] for row in task_rows}
    tasks = [int(task) for task in tasks]
    if not tasks or any(task not in task_names for task in tasks):
        raise ValueError("tasks must be within 0-{}".format(len(task_rows) - 1))
    suite_root = Path(output_root) / suite
    table = AttemptTable(suite_root / ATTEMPTS_CSV)
    demo_store = DemoStore(Path(demo_root) / suite, task_names) if save_demos else None
    videos_dir = suite_root / VIDEO_DIR
    result = CollectResult(0, "completed", None, suite_root, table.path)

    def pending_task() -> Optional[int]:
        return next((task for task in tasks if table.success_count(task) < demos_per_task), None)

    task_id = pending_task()
    if task_id is None:
        print("every requested {} task already has {} successes in {}".format(suite, demos_per_task, table.path))
        result.reason = "nothing to collect"
        return result

    enable_dpi_awareness()
    log = _CollectLog(log_root, suite)
    result.run_dir = log.path
    mapper = SixDofMapper(config)
    if device is None:
        # the startup neutral survives every new layout: the SpaceMouse works again right after a reset
        device = _NeutralInput() if synthetic_input else KeepNeutralSpaceMouse(config)
    mount = hand_mount
    spec = shadow_spec(mount)

    def make_session(task: int, hand_spec: LiberoTeleopSpec) -> CollectionSession:
        return CollectionSession(hand_spec, env_config, suite=suite, task_id=task, seed=seed,
                                 object_mass_scale=object_mass_scale, object_masses_kg=object_masses_kg)

    session = make_session(task_id, spec)
    keyboard = KeyboardCommandRouter()
    reset_feedback = ResetFeedback(keyboard)  # R during a reset / rebuild is dropped; the views say so
    commands: "queue.Queue[tuple[str, int]]" = queue.Queue()
    last_key_at: dict[str, float] = {}
    tri_view: Any = None
    camera: Any = None
    retargeter: Any = None
    sidecar: Any = None
    composer: Any = None
    scene: Any = None
    rays: Any = None
    hand_input: Any = None
    gestures: Any = None
    hand_resampler: Optional[FixedDelayResampler] = None
    hand_sample_seq = 0
    exit_code, reason = 0, "completed"
    last_buttons: tuple[bool, ...] = (False,) * 8
    loop_periods: list[float] = []
    clock = FixedRateClock(config.control_hz, high_resolution_timer=False)
    tri_config: Any = None
    monitor: Optional[TaskSuccessMonitor] = None
    disturb: Optional[DisturbanceMonitor] = None
    recorder: Optional[DemoRecorder] = None
    attempt: dict[str, Any] = {}
    other_objects: list[str] = []
    model_xml = ""
    camera_pose: dict[str, Any] = {}
    cached = {"saved": 0, "rate": "-"}
    run_attempts = 0
    rays_on = bool(rays_enabled)
    dt_sim = 0.0
    scene_version = 0
    scene_init: dict[str, Any] = {}
    scene_path: Optional[Path] = None
    scene_digest: Optional[str] = None
    planner_counts = {"requests": 0, "arrived": 0, "cancelled": 0, "failed": 0, "mount_switches": 0,
                      "scene_files": 0}
    video: Optional[AttemptVideo] = None  # the open attempt's video
    finishing: list[AttemptVideo] = []  # ended attempts' videos (written on their own threads)
    discarded: list[AttemptVideo] = []  # ended attempts' recordings without S (deleted on their own threads)
    layouts_used: dict[int, int] = {}  # highest layout index used this run, per task (R before a move too)

    def route(key: int, source: str) -> None:
        now = time.monotonic()
        if key in (ord("v"), ord("V")):
            if now - last_key_at.get("v", -10.0) > 2.0:
                print("the close-up is fixed in the collection entry (the V x2 view, mirrored)")
            last_key_at["v"] = now
            return
        if ord("0") <= key <= ord("9"):
            if now - last_key_at.get(chr(key), -10.0) < DIGIT_REPEAT_S:
                return
            last_key_at[chr(key)] = now
            commands.put(("cancel", 0) if key == ord("0") else ("target", key - ord("0")))
            return
        if key in (ord("l"), ord("L")):
            if now - last_key_at.get("l", -10.0) >= DIGIT_REPEAT_S:
                last_key_at["l"] = now
                commands.put(("rays", 0))
            return
        if key in (ord(MOUNT_SWITCH_KEY), ord(MOUNT_SWITCH_KEY.upper())):
            if now - last_key_at.get(MOUNT_SWITCH_KEY, -10.0) >= MOUNT_SWITCH_REPEAT_S:
                last_key_at[MOUNT_SWITCH_KEY] = now
                commands.put(("mount", 0))
            return
        if key in (ord(RECORD_KEY), ord(RECORD_KEY.upper())):
            if now - last_key_at.get(RECORD_KEY, -10.0) >= RECORD_REPEAT_S:
                commands.put(("record", 0))
            last_key_at[RECORD_KEY] = now  # every press, so a held key (auto-repeat) toggles once
            return
        keyboard.feed(key, source=source)

    def drop_stale_commands() -> None:
        while True:  # keys pressed while the scene was rebuilt
            try:
                commands.get_nowait()
            except queue.Empty:
                break

    def refresh_counts() -> None:
        cached["saved"] = table.success_count(task_id)
        cached["rate"] = table.stats(task_id).describe()

    settle_info: dict[str, Any] = {}

    def settle() -> None:
        """The objects this placement dropped land before the operator starts (not recorded)."""
        settle_info.clear()
        settle_info.update(settle_scene(env, composer))
        log.event(dict(settle_info, event="scene_settled", task_id=task_id, placement_seed=session.placement_seed))
        if not settle_info["settled"]:
            print("objects still moving after {:.1f} s of settling (fastest {:.3f} m/s); starting anyway".format(
                settle_info["sim_s"], settle_info["max_speed_mps"]))

    def save_scene(task_scene: PlannerScene) -> None:
        """Compile the model for the OMPL worker; kept while the model is unchanged (libero_object layouts)."""
        import mujoco

        nonlocal scene_version, scene_init, scene_path, scene_digest
        digest = hashlib.sha1(model_xml.encode("utf-8")).hexdigest()
        if digest == scene_digest and scene_path is not None and scene_path.is_file():
            return  # same compiled model: the worker keeps the scene it has loaded
        scene_version += 1
        SCENE_DIR.mkdir(parents=True, exist_ok=True)
        path = SCENE_DIR / "scene_{}_{}.mjb".format(os.getpid(), scene_version)
        mujoco.mj_saveModel(env.sim.model._model, str(path), None)
        _remove_quietly(scene_path)
        scene_path, scene_digest = path, digest
        scene_init = {"model_path": str(path), "arm_qpos_adr": task_scene.arm_qpos_adr,
                      "arm_dof_adr": task_scene.arm_dof_adr, "arm_joint_ids": task_scene.arm_joint_ids,
                      "eef_site": task_scene.eef_site, "robot_geoms": sorted(task_scene.robot_geoms)}
        planner_counts["scene_files"] += 1

    def build_placement() -> None:
        """Composer, targets, rays (and the planner's scene) for the new model, then settle."""
        nonlocal composer, scene, rays, model_xml
        model_xml = env.sim.model.get_xml()  # this placement's model (robosuite 1.4 rebuilds it on reset)
        composer = PlannerActionComposer(env, spec)
        composer.initialize_goal()
        scene = PlannerScene(env, composer)
        rays = FingertipRays(env, scene.robot_geoms, enabled=rays_on)
        if planner_backend == "ompl":
            save_scene(scene)
        settle()

    def rebind() -> None:
        """Inputs back to neutral, views on the new model (after a placement, M or a task switch)."""
        nonlocal tri_config
        mapper.reset()
        device.reset_input_state()
        hand_input.reset()
        hand_resampler.reset(discard_until_s=time.perf_counter())  # frames before the reset are stale
        gestures.reset(buttons=last_buttons)
        tri_view.track_site = composer.controller.eef_name
        tri_config = follow_robot_base(_shadow_tri_view_config(config), env)  # another task may move the base
        tri_view.config = tri_config
        tri_view.configure_cameras(env)
        mapper.set_camera_from_sim(env, tri_config.front_camera_name)
        drop_stale_commands()

    def bind_monitors() -> None:
        """This placement's success monitor, disturbance reference, recorder and main-view pose."""
        nonlocal monitor, disturb, recorder, other_objects, camera_pose
        monitor = TaskSuccessMonitor.for_env(env, hold_steps)
        other_objects = sorted(set(env.objects_dict) - goal_object_names(monitor.goal_states, env.objects_dict))
        disturb = DisturbanceMonitor(other_objects, DISTURB_MOVE_M, DISTURB_TILT_DEG)
        recorder = DemoRecorder(len(monitor.goal_states), hand_dof=spec.hand_dof)
        camera_pose = operator_camera(env, tri_config.front_camera_name)
        refresh_counts()

    def begin_attempt(placement_index: int) -> None:
        nonlocal run_attempts, video
        run_attempts += 1
        attempt.clear()
        attempt.update({
            "number": table.next_attempt_number(), "run_attempt": run_attempts, "placement_index": placement_index,
            "placement_seed": session.placement_seed, "started_at": _utc_now(),
            "started_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "wall_start": time.perf_counter(),
            "planner_events": [], "open": True,
            # the previous layout's last picture: not recorded if S comes before the window shows the new one
            "stale_frame": tri_view.last_frame if tri_view is not None else None,
        })
        layouts_used[task_id] = max(layouts_used.get(task_id, -1), int(placement_index))
        if video is not None:  # normally closed by end_attempt
            video.discard()
            discarded.append(video)
            video = None  # nothing is recorded until S

    def next_layout_index() -> int:
        """Past every layout this task used before (table) or this run (also ones left with R before a move)."""
        index = max(table.next_layout(task_id), layouts_used.get(task_id, -1) + 1)
        if demo_store is not None:
            index = max(index, demo_store.next_placement_index(task_id))
        return index

    def attempt_planner() -> dict[str, Any]:
        """This attempt's planner use: requests and how they ended, targets asked for, steps it drove the arm."""
        notes = attempt.get("planner_events", [])
        counts = Counter(note["event"] for note in notes)
        return {"requests": counts["request"], "arrived": counts["arrived"], "cancelled": counts["cancel"],
                "failed": counts["failed"], "targets": [note.get("target") for note in notes if note["event"] == "request"],
                "steps": recorder.planner_steps if recorder is not None else 0}

    def toggle_recording() -> None:
        """S: record the open attempt from now on (saved when it ends); S again: stop and delete that recording."""
        nonlocal video
        if not attempt.get("open"):
            return
        if video is None:
            video = AttemptVideo(
                videos_dir, fps=float(config.control_hz), title="TASK {}: {}".format(task_id, languages[task_id]),
                subtitle="{} | attempt {} | hand mount {}".format(suite, attempt["number"], mount),
                stale_frame=attempt.get("stale_frame"),
                work_dir=log.path,  # recorded in the run log; into videos/ only when the attempt ends
            )
        else:
            video.discard()  # ffmpeg stopped, the file deleted
            discarded.append(video)
            video = None
        log.event({"event": "video_record", "attempt": attempt["number"], "recording": video is not None,
                   "started": bool(recorder is not None and recorder.started),
                   "step": recorder.steps if recorder is not None else None, "timestamp_monotonic_s": time.monotonic()})
        print("attempt {}: {}".format(attempt["number"], "recording from now, saved when the attempt ends "
                                      "(S again: cancel)" if video is not None else "recording cancelled, nothing saved"))

    def end_attempt(why: str, *, success: bool = False, duration: Optional[float] = None,
                    demo_path: Optional[Path] = None) -> None:
        """Close the open attempt: its video (if S started one), its CSV row, the run log."""
        nonlocal video
        if not attempt.get("open") or recorder is None:
            return
        attempt["open"] = False
        outcome = "success" if success else attempt_outcome(recorder.started, why)
        now = time.perf_counter()
        time_s = duration if success else (
            round(recorder.elapsed_s(now - attempt["wall_start"]), 3) if recorder.started else None)
        ended_by = why if success else ended_by_text(why)
        flagged = disturb.flagged() if disturb else []
        clip, video = video, None
        video_name = ""
        if clip is not None:  # recording since S
            if outcome == "not_started":  # S before the first move, then R / M / Q: no file was started
                clip.discard()
                discarded.append(clip)
            else:
                video_name = attempt_video_name(suite, task_id, attempt["number"], outcome, time_s, task_names[task_id])
                clip.finish(outcome, time_s, ended_by, video_name)
                finishing.append(clip)
        if outcome != "not_started":
            written = table.append({
                "attempt": attempt["number"], "started_at": attempt["started_local"], "task_id": task_id,
                "task": languages[task_id], "hand_mount": mount, "layout": attempt["placement_index"],
                "result": outcome, "time_s": "" if time_s is None else "{:.2f}".format(time_s), "ended_by": ended_by,
                "moved_objects": "; ".join(describe_disturbance(entry) for entry in flagged),
                "video": video_name, "operator": operator,
            })
            if not written:
                print("attempts.csv is open in another program ({}); this row is kept and written later".format(
                    table.blocked), file=sys.stderr)
            if video_name:
                print("attempt {}: video -> {}".format(attempt["number"], videos_dir / video_name))
        record = {
            "attempt": attempt["number"], "task_id": task_id, "task_name": task_names[task_id],
            "hand_mount": mount, "placement": PLACEMENT, "placement_index": attempt["placement_index"],
            "placement_seed": attempt["placement_seed"], "outcome": outcome, "reason": why, "ended_by": ended_by,
            "demo_file": None if demo_path is None else demo_path.relative_to(demo_store.root).as_posix(),
            "duration_s": time_s, "total_s": round(now - attempt["wall_start"], 3), "steps": recorder.steps,
            "first_input_step": recorder.first_input_step, "planner": attempt_planner(), "disturbed_objects": flagged,
            "video": video_name or None, "run": log.run_id, "operator": operator,
            "started_at": attempt["started_at"], "ended_at": _utc_now(),
        }
        if demo_store is not None:
            demo_store.log_attempt(record)  # the offline builder's attempt list (--save-demos)
        log.event(dict(record, event="attempt_end", video_dropped_frames=clip.dropped if clip is not None else 0))
        refresh_counts()
        if outcome != "success":
            print("attempt {} ({} task {}): {} [{}]".format(record["attempt"], suite, task_id, outcome, ended_by))

    def handle_planner_events(tick: float) -> None:
        """Log, print and count the planner's events; the open attempt keeps them (step = next recorded step)."""
        for event in planner.drain_events():
            event["timestamp_monotonic_s"] = tick
            log.event(dict(event, task_id=task_id, attempt=attempt.get("number")))
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
            if attempt.get("open") and recorder is not None:
                attempt["planner_events"].append(planner_note(event, recorder.steps))

    def stop_planner(reason: str) -> None:
        """Before the scene changes or the program ends: a plan belongs to the current model."""
        if planner.cancel(reason) and composer is not None:
            composer.hold_here()
        handle_planner_events(time.monotonic())

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

    def next_placement() -> None:
        index = next_layout_index()
        session.reset_to(placement_seed(seed, task_id, index))
        build_placement()
        rebind()
        bind_monitors()
        begin_attempt(index)

    def open_task(task: int) -> None:
        nonlocal session, env, task_id
        task_id = task
        session = make_session(task_id, spec)
        session.open()
        env = session.env
        next_placement()
        print("targets:\n" + scene.table())

    def describe_targets() -> list[dict[str, Any]]:
        return [{"number": t.number, "name": t.name, "kind": t.kind} for t in scene.targets]

    def demo_attrs(final_duration: float, wall_duration: float, total_s: float) -> dict[str, Any]:
        robot = env.robots[0]
        model = env.sim.model._model
        return {
            "format": DEMO_FORMAT, "suite": suite, "task_id": task_id, "task_name": session.task.name,
            "language": session.task.language, "bddl_file": session.task.bddl_file,
            "problem_folder": session.task.problem_folder, "problem_info": session.problem_info,
            "env_kwargs": session.env_kwargs, "hand_mount": mount, "spec": spec.name,
            "gripper_type": spec.gripper_type, "action_dim": int(env.action_dim),
            "action_layout": {
                "arm": [0, 6], "hand": [6, int(env.action_dim)],
                "arm_meaning": "OSC_POSE delta to the adapter's desired pose: metres (world xyz), axis-angle radians",
                "hand_meaning": "Shadow actuator commands, normalized to the action limits",
            },
            "control_freq_hz": float(config.control_hz), "control_timestep_s": float(env.control_timestep),
            "model_timestep_s": float(env.model_timestep),
            "state_layout": {"time": 1, "qpos": int(model.nq), "qvel": int(model.nv), "act": int(model.na)},
            "success_rule": monitor.goal_states, "success_labels": monitor.labels,
            "success_hold_steps": monitor.hold.hold_steps, "success_check": "env._check_success()",
            "placement": PLACEMENT, "placement_seed": attempt["placement_seed"],
            "placement_index": attempt["placement_index"], "base_seed": seed,
            "first_input_step": recorder.first_input_step, "num_steps": recorder.steps, "settle": dict(settle_info),
            "duration_s": final_duration, "wall_duration_s": wall_duration, "total_s": total_s,
            "disturbed_objects": disturb.flagged(),
            "disturb_limits": {"move_m": DISTURB_MOVE_M, "tilt_deg": DISTURB_TILT_DEG},
            "arm_owner_codes": list(ARM_OWNERS),
            "planner": dict(attempt_planner(), backend=planner_backend, clearance_m=clearance_m,
                            max_speed_m_s=planner.max_speed_m_s,
                            max_angular_speed_rad_s=planner.max_angular_speed_rad_s, scene_targets=describe_targets()),
            "planner_events": list(attempt["planner_events"]),
            "arm_qpos_index": [int(i) for i in robot._ref_joint_pos_indexes],
            "hand_qpos_index": [int(i) for i in robot._ref_gripper_joint_pos_indexes],
            "eef_site": composer.controller.eef_name, "eef_body": robot.robot_model.eef_name,
            "object_bodies": {name: obj.root_body for name, obj in env.objects_dict.items()},
            "image_cameras": ["agentview", "robot0_eye_in_hand"], "operator_camera": camera_pose,
            "operator": operator, "attempt": attempt["number"], "run": log.run_id, "collected_at": _utc_now(),
            "teleop": {
                "arm_speed_scale": arm_speed_scale, "rotation_speed_scale": rotation_speed_scale,
                "osc_kp": session.osc_kp, "hand_control": hand_control, "camera_mirror": camera_mirror,
                "hand_target_delay_s": config.hand_target_delay_s,
            },
            "object_masses": session.object_masses, "versions": software_versions(),
        }

    env: Any = None
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
        index = next_layout_index()
        session.reset_to(placement_seed(seed, task_id, index))
        build_placement()
        if session.object_mass_scale != 1.0 or session.object_mass_overrides_kg:
            print(describe_object_masses(session))
        tri_config = follow_robot_base(_shadow_tri_view_config(config), env)

        def extras() -> list[Any]:
            return rays.geoms() if rays is not None else []

        def badge_points() -> list[tuple[str, np.ndarray]]:
            return scene.badge_points() if scene is not None else []

        tri_view = collection_view_window(
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
        bind_monitors()
        begin_attempt(index)
        log.metadata({
            "mode": "libero_data_collection", "base_entry": "libero_planner_teleop", "suite": suite, "tasks": tasks,
            "demos_per_task": demos_per_task, "hand_mount": mount, "mount_switch_key": MOUNT_SWITCH_KEY,
            "seed": seed, "placement": PLACEMENT, "success_hold_steps": hold_steps,
            "device": device_info, "hand_control": hand_control, "camera_mirror": camera_mirror,
            "hand_target_delay_ms": 1e3 * config.hand_target_delay_s,
            "dex": retargeter.metadata() if retargeter is not None else None,
            "sim_speed": {"requested": SIM_SPEED, "control_freq_hz": 1.0 / env.control_timestep,
                          "substeps": substeps, "sim_s_per_tick": dt_sim, "wall_hz": config.control_hz},
            "planner": {"backend": planner_backend, "clearance_m": clearance_m,
                        "max_speed_m_s": planner.max_speed_m_s,
                        "max_angular_speed_rad_s": planner.max_angular_speed_rad_s,
                        "ompl_time_limit_s": OMPL_TIME_LIMIT_S, "scene_dir": str(SCENE_DIR),
                        "python": str(discover_planner_python(planner_python)) if planner_backend == "ompl" else None,
                        "targets": describe_targets()},
            "arm": {
                "translation_speed_scale": arm_speed_scale, "rotation_speed_scale": rotation_speed_scale,
                "translation_scale_m": config.translation_scale_m, "max_translation_delta_m": config.max_translation_delta_m,
                "rotation_scale_rad": config.rotation_scale_rad, "max_rotation_delta_rad": config.max_rotation_delta_rad,
                "goal_leash_m": LIBERO_GOAL_LEASH_M, "goal_leash_rad": LIBERO_GOAL_LEASH_RAD, "osc_kp": session.osc_kp,
            },
            "rays": {"enabled": rays.enabled, "fingers": list(FINGERS), "direction": "world -z"},
            "close_up": {"azimuth_deg": COLLECT_DETAIL_FOLLOW.azimuth_deg, "fixed": True, "mirrored": True},
            "session": session.describe(), "views": tri_view.camera_setup,
            "control_frame": mapper.describe_control_frame(), "output": str(suite_root), "operator": operator,
            "disturb_limits": {"move_m": DISTURB_MOVE_M, "tilt_deg": DISTURB_TILT_DEG},
            "table": str(table.path),
            "video": {"folder": str(videos_dir), "source": "the view window as shown (webcam inset included)",
                      "fps": float(config.control_hz), "width": VIDEO_WIDTH, "crf": VIDEO_CRF,
                      "preset": VIDEO_PRESET, "preroll_s": PREROLL_S, "result_hold_s": RESULT_HOLD_S,
                      "recorded_from_key": RECORD_KEY},
            "save_demos": bool(save_demos), "demo_root": str(demo_store.root) if demo_store is not None else None,
        })
        print("LIBERO {} collection: tasks {} | {} demos each | hand mount {} ({}) | planner: {}, up to {:.2f} m/s".format(
            suite, ",".join(str(t) for t in tasks), demos_per_task, mount, MOUNT_TEXT[mount], planner_backend,
            planner.max_speed_m_s))
        print("task {} ({} of {} successes): {}".format(task_id, table.success_count(task_id), demos_per_task,
                                                        languages[task_id]))
        print("targets:\n" + scene.table())
        print("Keys (simulator window): 1-9 approach a target | 0 cancel | L fingertip rays | M switch hand mount "
              "(same layout) | S record this attempt from now | R new layout | Q quit. R / M count as a failure once "
              "the arm moved. Moving the SpaceMouse stops the planner and takes over.")
        print("Data: {} (attempts.csv + videos/; a video only for attempts where you pressed S)".format(suite_root))
        if demo_store is not None:
            print("Raw demos: {}".format(demo_store.root))
        started = previous = clock.restart()
        while duration_s is None or time.perf_counter() - started < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                stop_planner("reset")
                end_attempt("R")
                reset_feedback.begin(tri_view)
                next_placement()
                log.event({"event": "environment_reset", "task_id": task_id, "placement_seed": session.placement_seed,
                           "timestamp_monotonic_s": time.monotonic()})
                previous = clock.restart()
                continue
            mount_requested = False
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
                    rays_on = rays.enabled
                    print("fingertip rays {}".format("on" if rays.enabled else "off"))
                elif command == "record":
                    toggle_recording()
                elif command == "mount":
                    mount_requested = True
            if mount_requested:
                # M: the same task and layout again with the other hand mount.
                stop_planner("hand mount switch")
                end_attempt("M")
                reset_feedback.begin(tri_view, label="hand mount switch", notice=MOUNT_SWITCH_NOTICE)
                new_mount = other_mount(mount)
                print("switching hand mount: {} -> {} ({}); rebuilding the task ...".format(
                    mount, new_mount, MOUNT_TEXT[new_mount]))
                switch_started = time.perf_counter()
                layout = (attempt["placement_index"], attempt["placement_seed"])
                session.close()
                mount, spec = new_mount, shadow_spec(new_mount)
                session = make_session(task_id, spec)
                session.open()
                env = session.env
                session.reset_to(layout[1])
                build_placement()
                rebind()
                bind_monitors()
                begin_attempt(layout[0])
                planner_counts["mount_switches"] += 1
                log.event({"event": "hand_mount_switch", "hand_mount": mount, "spec": spec.name,
                           "rebuild_s": time.perf_counter() - switch_started, "targets": describe_targets(),
                           "timestamp_monotonic_s": time.monotonic()})
                print("hand mount now {} ({}), same layout, {:.1f} s".format(
                    mount, MOUNT_TEXT[mount], time.perf_counter() - switch_started))
                previous = clock.restart()
                continue
            loop_start = time.perf_counter()
            tick = time.monotonic()  # hand / button timestamps share the camera worker's clock
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
            if step_hook is not None:
                key = step_hook(env, SimpleNamespace(attempt=attempt["run_attempt"], step=recorder.steps,
                                                     task_id=task_id, mount=mount, planner=planner.state))
                if key:
                    route(ord(str(key)[0]), "hook")
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
            handle_planner_events(tick)
            state_before = flat_state(env)
            step_started = time.perf_counter()
            env.step(action)
            step_ms = 1e3 * (time.perf_counter() - step_started)
            moving = owner == "planner" or bool(np.any(wrist != 0.0))
            began = not recorder.started and moving
            confirmed = monitor.evaluate(env, recorder.started or moving)
            details = getattr(camera_command, "details", None) or {}
            recorder.append(
                state=state_before, action=action, success=monitor.success, goal_status=monitor.status,
                moving=moving, spacemouse_axes=sample.control_axes, spacemouse_buttons=last_buttons,
                arm_delta=wrist, hand_targets=hand_value, hand_valid=bool(camera_command.valid),
                mano_landmarks=details.get("mano_landmarks"), arm_owner=owner,
                wall_s=loop_start - attempt["wall_start"], sim_s=float(env.sim.data.time),
            )
            if began:  # the timer and the disturbance reference start with the first move
                disturb.set_reference(object_poses(env, other_objects))
                log.event({"event": "attempt_started", "attempt": attempt["number"], "step": recorder.first_input_step,
                           "by": owner, "timestamp_monotonic_s": tick})
            disturbed = disturb.update(object_poses(env, other_objects))
            elapsed = recorder.elapsed_s(loop_start - attempt["wall_start"])
            tri_view.panel_lines = tuple(collection_panel(
                task_id=task_id, language=languages[task_id], labels=monitor.labels, status=monitor.status,
                started=recorder.started, met_before_start=monitor.hold.met_before_start,
                hold_count=monitor.hold.count, hold_steps=monitor.hold.hold_steps, elapsed_s=elapsed,
                disturbed=disturbed, saved=cached["saved"], target=demos_per_task, rate_text=cached["rate"],
                recording=video is not None,
            ))
            visual = tri_view.update(env, status_text="COLLECT {} {} | demo {} of {} | mount {} | {} | hand={} | rays {}".format(
                suite, task_id, min(cached["saved"] + 1, demos_per_task), demos_per_task, mount, planner.status(),
                hand_state, "on" if rays.enabled else "off"))
            if video is not None:  # the window as shown; the timer runs from the first move
                video.push(tri_view.last_frame, elapsed if recorder.started else None)
            if visual.get("closed"):
                keyboard.feed("q", source="views")
            loop_periods.append(max(period, 1e-9))
            previous = loop_start
            log.event({
                "event": "step", "t": tick, "task_id": task_id, "attempt": attempt["number"], "step": recorder.steps - 1,
                "step_ms": step_ms, "busy_ms": 1e3 * (time.perf_counter() - loop_start),
                "hz": 1.0 / max(period, 1e-9), "started": recorder.started, "success": monitor.success,
                "hold": monitor.hold.count, "owner": owner, "planner": planner.state, "hand": hand_state,
                "hand_valid": bool(camera_command.valid), "wrist_norm": float(np.linalg.norm(wrist)),
                "spacemouse_armed": bool(sample.armed),
                "disturbed": [entry["object"] for entry in disturbed],
            })
            if reset_feedback.active:  # first step on the new model done: the scene is back
                log.event(reset_feedback.settle(tri_view))
            if monitor.mismatches == 1 and recorder.steps and not attempt.get("mismatch_logged"):
                attempt["mismatch_logged"] = True
                log.event({"event": "success_mismatch", "status": monitor.status, "official": monitor.success})
            if confirmed:
                stop_planner("success")  # still driving (rare): its path belongs to this layout
                final_duration = round((recorder.steps - recorder.first_input_step) / float(config.control_hz), 3)
                # The time shown and tabled is the timer the operator watched (wall clock since the first
                # move), the same for every result; the raw demo also keeps the step count's time.
                wall_duration = round(elapsed, 3)
                total_s = round(time.perf_counter() - attempt["wall_start"], 3)
                number = cached["saved"] + 1
                reset_feedback.begin(tri_view, label="success", notice="SUCCESS - {}/{} for this task ({:.1f} s)\n"
                                     "next layout loading - R is ignored until the scene is back".format(
                                         number, demos_per_task, wall_duration))
                path = None
                if demo_store is not None:  # --save-demos: the raw simulator demo for training later
                    path = demo_store.save_demo(task_id, recorder.arrays(flat_state(env)),
                                                demo_attrs(final_duration, wall_duration, total_s), model_xml)
                    result.demos.append(path)
                end_attempt("goal held {:.1f} s".format(monitor.hold.hold_steps / float(config.control_hz)),
                            success=True, duration=wall_duration, demo_path=path)
                result.successes += 1
                if task_id not in result.tasks_touched:
                    result.tasks_touched.append(task_id)
                flagged = disturb.flagged()
                print("SUCCESS task {} ({:.1f} s, {} steps{}){}{}".format(
                    task_id, wall_duration, recorder.steps,
                    ", planner {} steps".format(recorder.planner_steps) if recorder.planner_steps else "",
                    " | moved: " + ", ".join(describe_disturbance(entry) for entry in flagged) if flagged else "",
                    " | demo " + path.relative_to(demo_store.root).as_posix() if path is not None else ""))
                log.event({"event": "attempt_success", "demo": str(path) if path is not None else None,
                           "duration_s": wall_duration, "step_duration_s": final_duration,
                           "planner_steps": recorder.planner_steps,
                           "disturbed_objects": flagged, "timestamp_monotonic_s": time.monotonic()})
                if table.success_count(task_id) >= demos_per_task:
                    following = pending_task()
                    if following is None:
                        reason = "all tasks done"
                        tri_view.show_notice("ALL TASKS DONE - {} success{} this run\nclosing".format(
                            result.successes, "" if result.successes == 1 else "es"), wait_s=0.2)
                        if not headless:
                            time.sleep(ALL_DONE_SHOW_S)
                        break
                    tri_view.show_notice("TASK {} DONE - {}/{} successes\nloading task {}: {}".format(
                        task_id, demos_per_task, demos_per_task, following, languages[following]), wait_s=0.2)
                    print("task {} done; loading task {}: {}".format(task_id, following, languages[following]))
                    session.close()
                    open_task(following)
                    log.event({"event": "task_switch", "task_id": task_id, "targets": describe_targets(),
                               "timestamp_monotonic_s": time.monotonic()})
                else:
                    next_placement()
                previous = clock.restart()
                continue
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
            stop_planner("exit")
        except Exception as exc:  # the planner must not hide the shutdown
            print("could not stop the planner cleanly: {}".format(exc), file=sys.stderr)
        try:
            end_attempt(reason)  # an attempt still open: quit (or not_started) with the stop reason
        except Exception as exc:  # the attempt log must not hide the shutdown
            print("could not log the last attempt: {}".format(exc), file=sys.stderr)
        if video is not None:  # only if end_attempt failed: no half-written file
            video.discard()
            discarded.append(video)
        try:
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
            video_errors = []
            for clip in finishing:  # ended attempts' videos finish writing (normally well under a second)
                path = clip.wait(FINISH_TIMEOUT_S)
                if path is not None:
                    result.videos.append(path)
                elif clip.error or not clip.done:
                    video_errors.append(clip.error or "still writing after {:.0f} s".format(FINISH_TIMEOUT_S))
            for message in video_errors:
                print("a video could not be written: {}".format(message), file=sys.stderr)
            for clip in discarded:  # recordings without S: their files in the run log are deleted
                clip.wait(FINISH_TIMEOUT_S)
            unsaved = None
            if not table.flush():  # still open in Excel: keep the rows in a second file
                unsaved = table.save_unwritten(log.run_id)
                print("attempts.csv is still open in another program ({}); {} row{} written to {}".format(
                    table.blocked, table.pending, "" if table.pending == 1 else "s",
                    unsaved if unsaved is not None else "no file (that failed too)"), file=sys.stderr)
            loop_hz = 1.0 / float(np.median(loop_periods[1:])) if len(loop_periods) > 1 else None
            log.event({"event": "exit", "reason": reason, "median_loop_hz": loop_hz, "display": display,
                       "successes": result.successes, "videos": [str(path) for path in result.videos],
                       "video_errors": video_errors, "video_dropped_frames": sum(clip.dropped for clip in finishing),
                       "recordings_cancelled": sum(1 for clip in discarded if clip.started),
                       "demos": [str(path) for path in result.demos],
                       "table_unsaved": str(unsaved) if unsaved is not None else None,
                       "spacemouse_kept_neutral_resets": getattr(device, "kept_neutral_resets", None),
                       "planner": dict(planner_counts),
                       "clock": {"period_ms": 1e3 * clock.period_s, "ticks": clock.ticks,
                                 "overruns_dropped_ticks": clock.overruns},
                       "timestamp_monotonic_s": time.monotonic()})
            log.close()
        if loop_hz is not None:
            print("median control loop: {:.1f} Hz (target {:.0f} Hz)".format(loop_hz, config.control_hz))
        rows = [{"task_id": task, "task_name": task_names[task], "demos": table.success_count(task),
                 "target": demos_per_task, "stats": table.stats(task)} for task in tasks]
        print(format_summary(rows))
        print("LIBERO collection stopped: {}. This run: {} success{}, {} video{}. Data: {}".format(
            reason, result.successes, "" if result.successes == 1 else "es", len(result.videos),
            "" if len(result.videos) == 1 else "s", suite_root))
    result.exit_code, result.reason = exit_code, reason
    return result
