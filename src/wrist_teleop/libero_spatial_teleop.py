"""LIBERO Spatial teleoperation adapters for project-local Panda end effectors.

This module runs only with ``.venv-libero`` (Python 3.8 / robosuite 1.4).
It retains the official LIBERO Spatial BDDL. Its default-Panda saved state is
materialized in a reference environment, then copied by matching joint name
into the selected gripper model. The new gripper keeps its safe initial pose.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from .config import WristConfig
from .device import MotionSample, SpaceMouseHid
from .gripper_teleop import (
    ROBOTIQ_THREE_FINGER_GRIPPER_POLICY_TARGET_RATE_PER_S,
    ThreeFingerContinuousGripperButtonController,
)
from .hand_input import CameraHandWorker
from .mapper import SixDofMapper
from .runtime_controls import KeyboardCommandRouter
from .shadow_gripper import (
    RIGID_TCP_GRIPPER_TYPE as SHADOW_GRIPPER_TYPE,
    SHADOW_PALMAR_NORMAL_LOCAL,
    SHADOW_PALM_VIEW_CAMERA,
    ShadowHandRight,
    register_shadow_gripper,
)
from .shadow_dex import ShadowDexHandInput, ShadowDexRetargeter
from .shadow_hand_input import ShadowHandGestureController, ShadowLegacyHandInput
from .visualization import LaserPointer, TriViewCompositor


ROOT = Path(__file__).resolve().parents[2]
_LIBERO_SOURCE = ROOT / "third_party" / "LIBERO" / "libero" / "libero"

# LIBERO-only tuning. robosuite 1.4 re-anchors the OSC position goal to the
# *current* end-effector pose on every step, so the adapter keeps its own
# desired pose (the equivalent of robosuite 1.5's goal_update_mode="desired")
# and sends the bounded error to that pose as the OSC delta.
LIBERO_ARM_TRANSLATION_SPEED_SCALE = 2.5
LIBERO_ARM_ROTATION_SPEED_SCALE = 1.6
LIBERO_OSC_KP = 300.0
LIBERO_GOAL_LEASH_M = 0.05
LIBERO_GOAL_LEASH_RAD = 0.5
LIBERO_OSC_OUTPUT_LIMIT_M = 0.06
LIBERO_OSC_OUTPUT_LIMIT_RAD = 0.6
# 1270 + 10 px gutter + 640 = 1920 x 1080 composite; side column = 1/3 width.
LIBERO_MAIN_VIEW_WIDTH_PX = 1270
LIBERO_SIDE_VIEW_WIDTH_PX = 640


@dataclass(frozen=True)
class LiberoTeleopSpec:
    """One explicit official-Scene / project-hand contract."""

    name: str
    gripper_type: str
    hand_dof: int
    camera_hand: bool


PANDA_SHADOW_SPEC = LiberoTeleopSpec(
    name="panda_shadow", gripper_type=SHADOW_GRIPPER_TYPE, hand_dof=20, camera_hand=True
)
PANDA_ROBOTIQ_THREE_FINGER_SPEC = LiberoTeleopSpec(
    name="panda_robotiq_three_finger",
    gripper_type="RobotiqThreeFingerGripper",
    hand_dof=1,
    camera_hand=False,
)


class _NeutralInput:
    """Exact-zero synthetic device used only by explicit headless smokes."""

    def open(self) -> dict[str, Any]:
        return {"product_string": "synthetic-neutral-libero-smoke"}

    def reset_input_state(self) -> None:
        return None

    def poll(self) -> MotionSample:
        now = time.monotonic()
        zeros = np.zeros(6, dtype=np.float64)
        return MotionSample(
            zeros, (False,) * 8, True, True, now, buttons_valid=True,
            calibrated_axes=zeros.copy(), zero_bias_axes=zeros.copy(),
            neutral_locked=True, calibration_state="synthetic_neutral",
        )

    def close(self) -> None:
        return None


class _RunLog:
    def __init__(self, spec: LiberoTeleopSpec) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = ROOT / "outputs" / "libero" / f"{stamp}_{spec.name}_teleop"
        self.path.mkdir(parents=True, exist_ok=False)
        self._stream = (self.path / "telemetry.jsonl").open("w", encoding="utf-8")

    def metadata(self, contents: dict[str, Any]) -> None:
        (self.path / "metadata.json").write_text(
            json.dumps(contents, indent=2, default=_json_default), encoding="utf-8"
        )

    def event(self, contents: dict[str, Any]) -> None:
        self._stream.write(json.dumps(contents, default=_json_default, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("cannot serialize {!r}".format(type(value)))


def configure_local_libero_paths() -> None:
    """Configure bundled LIBERO sources without its first-import prompt."""

    config_root = ROOT / "outputs" / "libero" / ".libero_config"
    config_root.mkdir(parents=True, exist_ok=True)
    dataset_root = config_root / "datasets"
    dataset_root.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": str(_LIBERO_SOURCE),
        "bddl_files": str(_LIBERO_SOURCE / "bddl_files"),
        "init_states": str(_LIBERO_SOURCE / "init_files"),
        "datasets": str(dataset_root),
        "assets": str(_LIBERO_SOURCE / "assets"),
    }
    (config_root / "config.yaml").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    if os.name == "nt":
        os.environ["MUJOCO_GL"] = "wgl"


def ensure_windows_libero_runtime() -> None:
    """Supply only the isolated LIBERO venv's existing WGL / DLL shim."""

    if os.name != "nt":
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    source = site_packages / "mujoco" / "mujoco.dll"
    target = site_packages / "robosuite" / "utils" / "mujoco.dll"
    if not source.is_file():
        raise RuntimeError("LIBERO MuJoCo DLL is missing: {}".format(source))
    digest = lambda path: hashlib.sha256(path.read_bytes()).digest()
    if not target.is_file() or digest(target) != digest(source):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    macros_private = site_packages / "robosuite" / "macros_private.py"
    generated = (
        "# Generated by project LIBERO teleoperation adapter for Windows WGL support.\n"
        "import robosuite.macros as _robosuite_macros\n"
        "_robosuite_macros.MUJOCO_GPU_RENDERING = False\n"
    )
    if not macros_private.exists() or macros_private.read_text(encoding="utf-8").startswith(
        "# Generated by project LIBERO teleoperation adapter"
    ):
        macros_private.write_text(generated, encoding="utf-8")


def _joint_widths(model: Any, index: int) -> tuple[int, int]:
    next_joint = index + 1
    next_qpos = int(model.jnt_qposadr[next_joint]) if next_joint < model.njnt else int(model.nq)
    next_dof = int(model.jnt_dofadr[next_joint]) if next_joint < model.njnt else int(model.nv)
    return next_qpos - int(model.jnt_qposadr[index]), next_dof - int(model.jnt_dofadr[index])


def _copy_matching_joint_state(source: Any, target: Any) -> list[str]:
    """Copy exact same-shape named joints; never copy incompatible hand state."""

    source_model, source_data = source.sim.model, source.sim.data
    target_model, target_data = target.sim.model, target.sim.data
    copied: list[str] = []
    target_names = set(target_model.joint_names)
    for name in source_model.joint_names:
        if name is None or name not in target_names:
            continue
        source_id, target_id = int(source_model.joint_name2id(name)), int(target_model.joint_name2id(name))
        source_qwidth, source_dwidth = _joint_widths(source_model, source_id)
        target_qwidth, target_dwidth = _joint_widths(target_model, target_id)
        if source_qwidth != target_qwidth or source_dwidth != target_dwidth:
            continue
        source_qpos, target_qpos = int(source_model.jnt_qposadr[source_id]), int(target_model.jnt_qposadr[target_id])
        source_dof, target_dof = int(source_model.jnt_dofadr[source_id]), int(target_model.jnt_dofadr[target_id])
        target_data.qpos[target_qpos:target_qpos + target_qwidth] = source_data.qpos[source_qpos:source_qpos + source_qwidth]
        target_data.qvel[target_dof:target_dof + target_dwidth] = source_data.qvel[source_dof:source_dof + source_dwidth]
        copied.append(str(name))
    return copied


class OfficialLiberoSpatialSession:
    """One official Spatial task plus one selected non-default hand."""

    def __init__(
        self,
        spec: LiberoTeleopSpec,
        config: WristConfig,
        *,
        task_id: int,
        task_order: int,
        init_state_id: int,
        seed: int,
        has_offscreen_renderer: bool,
    ) -> None:
        if task_id < 0 or task_order < 0 or init_state_id < 0:
            raise ValueError("task-order, task-id, and init-state-id must be nonnegative")
        self.spec, self.config = spec, config
        self.task_id, self.task_order, self.init_state_id, self.seed = int(task_id), int(task_order), int(init_state_id), int(seed)
        self.has_offscreen_renderer = bool(has_offscreen_renderer)
        self.reference: Any = None
        self.env: Any = None
        self.task: Any = None
        self.init_states: Any = None
        self._copied_joint_names: list[str] = []

    def _controller(self) -> dict[str, Any]:
        import robosuite as suite

        controller = copy.deepcopy(suite.load_controller_config(default_controller="OSC_POSE"))
        maximums = [LIBERO_OSC_OUTPUT_LIMIT_M] * 3 + [LIBERO_OSC_OUTPUT_LIMIT_RAD] * 3
        # Legacy robosuite maps action input to output. Equal ranges make the
        # arm slice a physical metre / radian error to the adapter's desired
        # pose. The SpaceMouse mapper still bounds each per-step increment.
        controller.update(
            input_min=[-value for value in maximums], input_max=maximums,
            output_min=[-value for value in maximums], output_max=maximums,
            control_delta=True, impedance_mode="fixed", kp=LIBERO_OSC_KP,
        )
        return controller

    def open(self) -> None:
        configure_local_libero_paths()
        ensure_windows_libero_runtime()
        if self.spec == PANDA_SHADOW_SPEC:
            register_shadow_gripper()
        try:
            import torch
            from libero.libero import get_libero_path
            from libero.libero.benchmark import get_benchmark
            from libero.libero.envs import OffScreenRenderEnv, bddl_utils
            from libero.libero.envs.bddl_base_domain import TASK_MAPPING
        except ImportError as exc:
            raise RuntimeError("official LIBERO dependencies are unavailable in .venv-libero") from exc
        benchmark = get_benchmark("libero_spatial")(self.task_order)
        if self.task_id >= benchmark.get_num_tasks():
            raise ValueError("task-id {} is outside official libero_spatial range".format(self.task_id))
        self.task = benchmark.get_task(self.task_id)
        bddl_path = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        init_path = Path(get_libero_path("init_states")) / self.task.problem_folder / self.task.init_states_file
        self.init_states = torch.load(init_path, map_location="cpu")
        if self.init_state_id >= len(self.init_states):
            raise ValueError("init-state-id {} is outside official task state range".format(self.init_state_id))
        common = dict(
            # Teleoperation never reads policy image observations. Rendering
            # them on every step cost three extra offscreen passes; the
            # project three-view renders its own cameras from the same state.
            bddl_file_name=str(bddl_path), robots=["Panda"], use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=True,
            control_freq=int(round(self.config.control_hz)), horizon=10_000_000,
            ignore_done=True, camera_names=list(self.config.tri_view.camera_names),
            camera_heights=128, camera_widths=128, initialization_noise=None,
        )
        # Untouched official default Panda model: it only materializes the
        # task's released initial state before compatible state transfer.
        self.reference = OffScreenRenderEnv(**common)
        info = bddl_utils.get_problem_info(str(bddl_path))
        task_class = TASK_MAPPING[info["problem_name"]]
        self.env = task_class(
            **common, controller_configs=self._controller(), gripper_types=self.spec.gripper_type,
        )
        self.env.seed(self.seed)
        self.reset()

    def reset(self) -> Any:
        if self.reference is None or self.env is None or self.init_states is None:
            raise RuntimeError("open the LIBERO session before reset")
        self.reference.reset()
        self.reference.set_init_state(self.init_states[self.init_state_id])
        self.env.reset()
        self._copied_joint_names = _copy_matching_joint_state(self.reference, self.env)
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        return self.env._get_observations()

    @property
    def copied_joint_names(self) -> list[str]:
        return list(self._copied_joint_names)

    def describe(self) -> dict[str, Any]:
        if self.env is None or self.task is None:
            raise RuntimeError("session is not open")
        lower, upper = (np.asarray(value, dtype=np.float64) for value in self.env.action_spec)
        return {
            "suite": "libero_spatial", "task_id": self.task_id, "task_order": self.task_order,
            "task_name": self.task.name, "language": self.task.language,
            "init_state_id": self.init_state_id, "seed": self.seed,
            "bddl_file": self.task.bddl_file, "gripper_type": self.spec.gripper_type,
            "action_dim": int(self.env.action_dim), "action_low": lower.tolist(), "action_high": upper.tolist(),
            "initial_state_strategy": "official_default_state_then_matching_named_joint_transfer",
            "copied_joint_count": len(self._copied_joint_names), "copied_joint_names": self.copied_joint_names,
        }

    def close(self) -> None:
        for env in (self.env, self.reference):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        self.env, self.reference = None, None


class LiberoActionComposer:
    """Compose six physical OSC deltas and the selected real hand contract."""

    def __init__(self, env: Any, spec: LiberoTeleopSpec) -> None:
        self.env, self.spec, self.robot = env, spec, env.robots[0]
        low, high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
        expected = 6 + spec.hand_dof
        if low.shape != (expected,) or high.shape != (expected,) or int(env.action_dim) != expected:
            raise RuntimeError("LIBERO action contract does not equal six OSC plus selected hand dimensions")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or not np.all(high > low):
            raise RuntimeError("LIBERO action limits are invalid")
        self.low, self.high = low, high
        self.arm_slice, self.hand_slice = slice(0, 6), slice(6, expected)
        self.gripper = self.robot.gripper
        if int(getattr(self.gripper, "dof", -1)) != spec.hand_dof:
            raise RuntimeError("loaded gripper action count does not match selected adapter")
        if spec == PANDA_SHADOW_SPEC and not isinstance(self.gripper, ShadowHandRight):
            raise RuntimeError("LIBERO Shadow entry did not load the project-local Shadow hand")
        self.controller = self.robot.controller
        self.goal_pos: Optional[np.ndarray] = None
        self.goal_ori: Optional[np.ndarray] = None
        self._moving = False

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the exact site pose robosuite 1.4 OSC uses in ``set_goal``."""
        sim = self.env.sim
        site = sim.model.site_name2id(self.controller.eef_name)
        return (
            np.array(sim.data.site_xpos[site], dtype=np.float64),
            np.array(sim.data.site_xmat[site], dtype=np.float64).reshape(3, 3),
        )

    def initialize_goal(self) -> None:
        reset_goal = getattr(self.controller, "reset_goal", None)
        if callable(reset_goal):
            reset_goal()
        self.goal_pos, self.goal_ori = self.ee_pose()

    def zero_action(self) -> np.ndarray:
        return np.zeros(int(self.env.action_dim), dtype=np.float64)

    def arm_action(self, wrist: np.ndarray) -> np.ndarray:
        """Integrate one world delta into the desired pose; return the OSC error.

        Zero input keeps the same desired pose, so finger reaction forces and
        gravity sag are corrected instead of being re-latched every step. The
        desired pose is leashed to the measured pose so a blocked arm cannot
        wind up a large hidden error.
        """
        from robosuite.utils import transform_utils as T

        position, orientation = self.ee_pose()
        moving = bool(np.any(wrist != 0.0))
        if self.goal_pos is None or self.goal_ori is None or (self._moving and not moving):
            # Release edge: stop where the arm is instead of letting it catch
            # up to a desired pose that was leading it during fast motion.
            self.goal_pos, self.goal_ori = position.copy(), orientation.copy()
        self._moving = moving
        self.goal_pos = self.goal_pos + wrist[:3]
        if np.any(wrist[3:] != 0.0):
            self.goal_ori = T.quat2mat(T.axisangle2quat(wrist[3:])) @ self.goal_ori
        position_error = self.goal_pos - position
        distance = float(np.linalg.norm(position_error))
        if distance > LIBERO_GOAL_LEASH_M:
            position_error *= LIBERO_GOAL_LEASH_M / distance
            self.goal_pos = position + position_error
        rotation_error = np.asarray(T.quat2axisangle(T.mat2quat(self.goal_ori @ orientation.T)), dtype=np.float64)
        angle = float(np.linalg.norm(rotation_error))
        if angle > LIBERO_GOAL_LEASH_RAD:
            rotation_error *= LIBERO_GOAL_LEASH_RAD / angle
            self.goal_ori = T.quat2mat(T.axisangle2quat(rotation_error)) @ orientation
        error = np.concatenate((position_error, rotation_error))
        if not np.all(np.isfinite(error)):
            self.goal_pos, self.goal_ori = position.copy(), orientation.copy()
            error = np.zeros(6, dtype=np.float64)
        return np.clip(error, self.low[self.arm_slice], self.high[self.arm_slice])

    def compose(self, wrist_delta: Sequence[float], hand_value: Any = None) -> np.ndarray:
        try:
            wrist = np.asarray(wrist_delta, dtype=np.float64)
        except (TypeError, ValueError):
            wrist = np.zeros(6, dtype=np.float64)
        if wrist.shape != (6,) or not np.all(np.isfinite(wrist)):
            wrist = np.zeros(6, dtype=np.float64)
        action = self.zero_action()
        action[self.arm_slice] = self.arm_action(wrist)
        if self.spec == PANDA_SHADOW_SPEC:
            targets = np.zeros(20, dtype=np.float64) if hand_value is None else np.asarray(hand_value, dtype=np.float64)
            if targets.shape != (20,) or not np.all(np.isfinite(targets)):
                targets = np.zeros(20, dtype=np.float64)
            policy = np.asarray(self.gripper.targets_to_action(targets), dtype=np.float64)
            if policy.shape != (20,):
                raise RuntimeError("Shadow target conversion returned an invalid action")
            action[self.hand_slice] = np.clip(policy, self.low[self.hand_slice], self.high[self.hand_slice])
        else:
            try:
                command = float(0.0 if hand_value is None else hand_value)
            except (TypeError, ValueError, OverflowError):
                command = 0.0
            action[self.hand_slice] = np.clip(command if np.isfinite(command) else 0.0, self.low[self.hand_slice], self.high[self.hand_slice])
        return action


def _shadow_tri_view_config(config: WristConfig) -> Any:
    from dataclasses import replace

    return replace(
        config.tri_view,
        # robosuite 1.4 applies the generic ``gripper0_`` prefix when it
        # merges the project-local hand; 1.5 uses ``gripper0_right_``. The
        # adapter deliberately uses the actual legacy merge name here.
        wrist_camera_name="gripper0_{}".format(SHADOW_PALM_VIEW_CAMERA),
        wrist_fovy_deg=82.0,
    )


def _shadow_laser_config(config: WristConfig) -> Any:
    from dataclasses import replace

    return replace(
        config.laser,
        start_site="palm_laser_site", direction_frame="site",
        local_direction=SHADOW_PALMAR_NORMAL_LOCAL, visible_views=("front", "side"),
    )


class _WindowsTimerResolution:
    """Request 1 ms Windows timer ticks while the control loop runs.

    Python 3.8 ``time.sleep`` rounds up to the default 15.6 ms system tick,
    which alone cost ~14 ms per 50 ms control period.
    """

    def __init__(self) -> None:
        self._active = False

    def __enter__(self) -> "_WindowsTimerResolution":
        if os.name == "nt":
            try:
                import ctypes

                self._active = ctypes.windll.winmm.timeBeginPeriod(1) == 0
            except Exception:
                self._active = False
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._active:
            import ctypes

            ctypes.windll.winmm.timeEndPeriod(1)
            self._active = False


def _sleep(start: float, control_hz: float) -> None:
    remaining = 1.0 / float(control_hz) - (time.monotonic() - start)
    if remaining > 0.0:
        time.sleep(remaining)


def run_libero_spatial_teleop(
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
    three_view: bool,
    three_finger_rate_per_s: Optional[float] = None,
    arm_speed_scale: float = LIBERO_ARM_TRANSLATION_SPEED_SCALE,
    rotation_speed_scale: float = LIBERO_ARM_ROTATION_SPEED_SCALE,
) -> int:
    """Run one released LIBERO Spatial state through the existing controls."""

    if spec == PANDA_SHADOW_SPEC and hand_control not in ("dex", "legacy-curl", "off"):
        raise ValueError("LIBERO Shadow accepts dex, legacy-curl, or off")
    if spec == PANDA_ROBOTIQ_THREE_FINGER_SPEC and hand_control != "off":
        raise ValueError("LIBERO RobotiqThreeFinger has no camera retargeting; use hand-control off")
    rate = ROBOTIQ_THREE_FINGER_GRIPPER_POLICY_TARGET_RATE_PER_S if three_finger_rate_per_s is None else float(three_finger_rate_per_s)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("three-finger rate must be positive")

    for name, value in (("arm speed", arm_speed_scale), ("rotation speed", rotation_speed_scale)):
        if not np.isfinite(value) or not 0.1 <= float(value) <= 5.0:
            raise ValueError("{} scale must be in [0.1, 5]".format(name))
    # Same mapper pipeline (deadzone, EMA, clamp, camera frame); only the
    # per-step physical scale is raised for this entry.
    config = replace(
        config,
        translation_scale_m=config.translation_scale_m * arm_speed_scale,
        max_translation_delta_m=config.max_translation_delta_m * arm_speed_scale,
        rotation_scale_rad=config.rotation_scale_rad * rotation_speed_scale,
        max_rotation_delta_rad=config.max_rotation_delta_rad * rotation_speed_scale,
    )
    log = _RunLog(spec)
    mapper = SixDofMapper(config)
    device: Any = _NeutralInput() if synthetic_input else SpaceMouseHid(config)
    session = OfficialLiberoSpatialSession(
        spec, config, task_id=task_id, task_order=task_order, init_state_id=init_state_id,
        seed=seed, has_offscreen_renderer=bool(three_view),
    )
    keyboard = KeyboardCommandRouter()
    tri_view = laser = camera = retargeter = None
    composer = hand_input = gestures = gripper_buttons = None
    exit_code, reason = 0, "completed"
    last_buttons: tuple[bool, ...] = (False,) * 8
    screenshot_written = False
    loop_periods: list[float] = []

    def route(key: int, source: str) -> None:
        keyboard.feed(key, source=source)

    timer_resolution = _WindowsTimerResolution().__enter__()
    try:
        device_info = device.open()
        session.open()
        env = session.env
        assert env is not None
        composer = LiberoActionComposer(env, spec)
        composer.initialize_goal()
        tri_config = _shadow_tri_view_config(config) if spec == PANDA_SHADOW_SPEC else config.tri_view
        # Show the native 1920x1080 composite 1:1. A HighGUI-scaled window
        # was both blurrier and ~15 ms slower per frame. The right-hand
        # column (palm view over side view) takes one third of the width.
        tri_config = replace(
            tri_config,
            image_width=LIBERO_MAIN_VIEW_WIDTH_PX,
            thumbnail_scale=LIBERO_SIDE_VIEW_WIDTH_PX / LIBERO_MAIN_VIEW_WIDTH_PX,
            display_width=None, display_height=None,
        )
        if three_view and tri_config.enabled:
            laser_config = _shadow_laser_config(config) if spec == PANDA_SHADOW_SPEC else config.laser
            laser = LaserPointer(env, laser_config)
            tri_view = TriViewCompositor(
                tri_config, show_window=not headless,
                key_handler=lambda key: route(key, "tri_view"),
                on_close=lambda: keyboard.feed("q", source="tri_view_window"), laser=laser,
            )
            tri_view.configure_cameras(env)
            mapper.set_camera_from_sim(env, tri_config.front_camera_name)
        if spec == PANDA_SHADOW_SPEC:
            if hand_control == "dex":
                print("Starting official dex-retargeting sidecar for Shadow (first launch can take ~60 s)...")
                retargeter = ShadowDexRetargeter()
                hand_input = ShadowDexHandInput(
                    retargeter, composer.gripper.lower_limits, composer.gripper.upper_limits,
                    stale_timeout_s=config.dex_hand.stale_timeout_s,
                    expected_handedness=config.dex_hand.expected_handedness,
                    # The worker flips frames for MediaPipe; dex expects the
                    # unflipped hand geometry used by the official detector.
                    unmirror_world_x=camera_mirror,
                )
                print("hand mode=dex | dex-retargeting={} | runtime=sidecar | DexPilot".format(
                    retargeter.metadata()["version"]))
            else:
                hand_input = ShadowLegacyHandInput(composer.gripper.lower_limits, composer.gripper.upper_limits)
            gestures = ShadowHandGestureController(
                config=config.hand_gestures,
                lower_limits=composer.gripper.lower_limits, upper_limits=composer.gripper.upper_limits,
            )
            gestures.reset(buttons=last_buttons)
            if hand_control in ("dex", "legacy-curl"):
                camera = CameraHandWorker(
                    hand_input, camera_index=camera_index,
                    display=camera_display and not headless, mirror_input=camera_mirror,
                    keypress_callback=lambda key: route(key, "camera"),
                )
                camera.start()
        else:
            gripper_buttons = ThreeFingerContinuousGripperButtonController(
                config,
                open_command=float(composer.low[composer.hand_slice][0]),
                close_command=float(composer.high[composer.hand_slice][0]), rate_per_s=rate,
            )
        log.metadata({
            "mode": "official_libero_spatial_teleoperation", "spec": spec.name,
            "session": session.describe(), "device": device_info,
            "hand_control": hand_control, "camera_mirror": camera_mirror,
            "dex": retargeter.metadata() if retargeter is not None else None,
            "arm": {
                "translation_speed_scale": arm_speed_scale, "rotation_speed_scale": rotation_speed_scale,
                "translation_scale_m": config.translation_scale_m, "max_translation_delta_m": config.max_translation_delta_m,
                "rotation_scale_rad": config.rotation_scale_rad, "max_rotation_delta_rad": config.max_rotation_delta_rad,
                "goal_mode": "adapter_desired_pose_with_leash", "osc_kp": LIBERO_OSC_KP,
                "goal_leash_m": LIBERO_GOAL_LEASH_M, "goal_leash_rad": LIBERO_GOAL_LEASH_RAD,
                "osc_tcp": composer.gripper.describe().get("osc_tcp_parent") if spec == PANDA_SHADOW_SPEC else "gripper grip_site",
            },
            "three_finger_rate_per_s": rate if gripper_buttons is not None else None,
            "control_frame": mapper.describe_control_frame(),
            "initial_state_note": "Official BDDL and task objects are unchanged; only compatible named state is transferred because the non-default hand has a different MuJoCo state length.",
        })
        print("LIBERO Spatial ready: {} | {}".format(spec.name, session.task.language))
        started = previous = time.monotonic()
        while duration_s is None or time.monotonic() - started < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                session.reset()
                composer = LiberoActionComposer(env, spec)
                composer.initialize_goal()
                mapper.reset()
                device.reset_input_state()
                if hand_input is not None:
                    hand_input.reset()
                if gestures is not None:
                    gestures.reset(buttons=last_buttons)
                if gripper_buttons is not None:
                    gripper_buttons.reset()
                if tri_view is not None:
                    # robosuite 1.4 hard reset destroys the old MjSim; the
                    # laser must bind to the new model (the old one cannot
                    # even be restored, so it is simply dropped).
                    laser = LaserPointer(env, laser_config)
                    tri_view.laser = laser
                    tri_view.configure_cameras(env)
                    mapper.set_camera_from_sim(env, tri_config.front_camera_name)
                log.event({"event": "environment_reset", "timestamp_monotonic_s": time.monotonic()})
                previous = time.monotonic()
                continue
            tick = time.monotonic()
            sample = device.poll()
            last_buttons = sample.buttons if sample.buttons_valid else (False,) * len(sample.buttons)
            wrist = mapper.map(sample.control_axes, valid=bool(sample.valid and sample.armed))
            # The mapper emits a per-control-period increment. If rendering or
            # the camera slows the loop, scale by the real period so the arm's
            # wall-clock speed does not fall with the frame rate.
            wrist = wrist * float(np.clip((tick - previous) * config.control_hz, 1.0, 2.0))
            if spec == PANDA_SHADOW_SPEC:
                assert hand_input is not None and gestures is not None
                camera_command = hand_input.latest(tick)
                hand_command = gestures.update(
                    last_buttons, camera_command.joint_targets,
                    camera_valid=bool(camera_command.valid and hand_control in ("dex", "legacy-curl")), timestamp_s=tick,
                )
                hand_value, hand_state = hand_command.joint_targets, hand_command.source
                hand_details = {
                    "source": hand_state, "camera_valid": camera_command.valid,
                    "camera_tracking_state": camera_command.tracking_state,
                    "camera_error": camera_command.error, "camera_details": camera_command.details,
                }
            else:
                assert gripper_buttons is not None
                button_command = gripper_buttons.update(
                    last_buttons, valid=bool(sample.buttons_valid and sample.armed),
                    dt_s=max(tick - previous, 0.0),
                )
                hand_value, hand_state = button_command.value, button_command.state
                hand_details = {
                    "state": button_command.state, "rising_edges": button_command.rising_edges,
                    "policy_target": gripper_buttons.target,
                }
            action = composer.compose(wrist, hand_value)
            _obs, reward, _done, _info = env.step(action)
            laser_record = laser.update() if laser is not None else {"enabled": False}
            visual = tri_view.update(
                env,
                status_text="LIBERO Spatial | {} | hand={} | wrist={:.4f}".format(
                    spec.name, hand_state, float(np.linalg.norm(wrist))
                ),
            ) if tri_view is not None else {"enabled": False}
            if tri_view is not None and visual.get("rendered") and not screenshot_written:
                screenshot = log.path / "tri_view_initial.png"
                visual["screenshot"] = str(screenshot)
                screenshot_written = bool(tri_view.save_last_frame(screenshot))
            if tri_view is not None and visual.get("closed"):
                keyboard.feed("q", source="tri_view")
            camera_display_record = camera.show_latest() if (
                camera is not None and camera_display and not headless and hasattr(camera, "show_latest")
            ) else {"enabled": False}
            elapsed = max(tick - previous, 1e-9)
            loop_periods.append(elapsed)
            previous = tick
            log.event({
                "event": "step", "timestamp_monotonic_s": tick,
                "input_valid": sample.valid, "input_armed": sample.armed, "input_error": sample.error,
                "buttons": sample.buttons, "arm_action": action[composer.arm_slice],
                "hand_action": action[composer.hand_slice], "mapped_world_delta": wrist,
                "hand": hand_details, "visualization": {"laser": laser_record, "tri_view": visual},
                "camera_display": camera_display_record, "reward": reward,
                "task_success": bool(env._check_success()), "update_hz": 1.0 / elapsed,
            })
            _sleep(tick, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        exit_code, reason = 1, "error: {}: {}".format(type(exc).__name__, exc)
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
    finally:
        try:
            if composer is not None and session.env is not None:
                final = composer.compose(np.zeros(6), None)
                session.env.step(final)
                log.event({"event": "safe_zero", "action": final, "timestamp_monotonic_s": time.monotonic()})
            if tri_view is not None:
                tri_view.close()
            if laser is not None:
                laser.close()
            if camera is not None:
                camera.close()
            if retargeter is not None:
                retargeter.close()
            mapper.reset()
            device.close()
            session.close()
        finally:
            timer_resolution.__exit__()
            loop_hz = 1.0 / float(np.median(loop_periods[1:])) if len(loop_periods) > 1 else None
            log.event({"event": "exit", "reason": reason, "median_loop_hz": loop_hz,
                       "timestamp_monotonic_s": time.monotonic()})
            log.close()
        if loop_hz is not None:
            print("median control/render loop: {:.1f} Hz (target {:.0f} Hz)".format(loop_hz, config.control_hz))
        print("LIBERO teleoperation stopped: {}. Log: {}".format(reason, log.path))
    return exit_code
