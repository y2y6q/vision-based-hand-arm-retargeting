"""Shared one-degree-of-freedom gripper teleoperation for robosuite.

The module deliberately keeps Panda's parallel grippers and the official
Jaco / Robotiq three-finger grippers on the same control path. It discovers the installed
OSC and policy-gripper action slices from the live composite controller rather
than assuming either layout is ``[0:6] + [6]``.

Interactive runs use the existing SpaceMouse HID reader and six-axis mapper.
For two-finger grippers, the two SpaceMouse buttons select open / close on a
rising edge. For three-finger grippers they are level-triggered: hold left to
close and hold right to open. Invalid / disconnected HID packets fail closed
to a zero wrist delta and zero gripper command.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .config import WristConfig
from .device import MotionSample, SpaceMouseHid
from .grasp_telemetry import GraspTelemetry
from .mapper import SixDofMapper
from .osc import DesiredGoalGuard
from .interactive_cube import InteractiveCubeScaler
from .recording import SimVideoRecorder
from .runtime_controls import KeyboardCommandRouter, install_viewer_key_callback


ROOT = Path(__file__).resolve().parents[2]

# Three-finger grippers consume a normalized velocity policy command.  The
# Panda + Robotiq and Jaco both reach the legal full-close / full-open command
# in 0.10 s at 20 Hz.  Neither setting exceeds the official action range;
# MuJoCo joint limits and the installed controller remain authoritative.
THREE_FINGER_GRIPPER_POLICY_TARGET_RATE_PER_S = 3.0
ROBOTIQ_THREE_FINGER_GRIPPER_POLICY_TARGET_RATE_PER_S = 12.0
JACO_GRIPPER_POLICY_TARGET_RATE_PER_S = 12.0
# robosuite's JacoThreeFingerGripper.format_action runs once per policy step,
# uses only sign(action), and moves its finger target by a fixed 0.01: a full
# -1 -> +1 sweep takes 200 steps (10 s at 20 Hz).  The Jaco entry raises that
# per-step travel and makes it follow the command magnitude, so a held button
# sweeps fully in about 0.7 s and the button ramp still eases the start.
JACO_GRIPPER_TARGET_SPEED_PER_STEP = 0.15

# Default configurations retained for the two original launchers.
ROBOT_GRIPPER_EXPECTATIONS = {
    "Panda": "PandaGripper",
    "Jaco": "JacoThreeFingerGripper",
}


@dataclass(frozen=True)
class GripperTeleopSpec:
    """One tested robosuite robot / official one-DoF gripper pairing."""

    robot: str
    gripper_type: str
    fingers: int
    continuous_buttons: bool
    continuous_rate_per_s: float | None = None

    def __post_init__(self) -> None:
        if self.continuous_buttons:
            if self.continuous_rate_per_s is None or not np.isfinite(self.continuous_rate_per_s):
                raise ValueError("continuous three-finger grippers require a finite button rate")
            if self.continuous_rate_per_s <= 0.0:
                raise ValueError("continuous three-finger gripper button rate must be positive")
        elif self.continuous_rate_per_s is not None:
            raise ValueError("edge-triggered grippers must not define a continuous button rate")


GRIPPER_TELEOP_SPECS: tuple[GripperTeleopSpec, ...] = (
    GripperTeleopSpec("Panda", "PandaGripper", fingers=2, continuous_buttons=False),
    GripperTeleopSpec("Panda", "Robotiq85Gripper", fingers=2, continuous_buttons=False),
    GripperTeleopSpec(
        "Jaco", "JacoThreeFingerGripper", fingers=3, continuous_buttons=True,
        continuous_rate_per_s=JACO_GRIPPER_POLICY_TARGET_RATE_PER_S,
    ),
    GripperTeleopSpec(
        "Panda", "RobotiqThreeFingerGripper", fingers=3, continuous_buttons=True,
        continuous_rate_per_s=ROBOTIQ_THREE_FINGER_GRIPPER_POLICY_TARGET_RATE_PER_S,
    ),
)


def resolve_gripper_spec(robot: str, gripper_type: str | None = None) -> GripperTeleopSpec:
    """Resolve only a pair exercised against the installed robosuite API."""

    expected = gripper_type or ROBOT_GRIPPER_EXPECTATIONS.get(robot)
    for spec in GRIPPER_TELEOP_SPECS:
        if spec.robot == robot and spec.gripper_type == expected:
            return spec
    supported = ", ".join(f"{item.robot}+{item.gripper_type}" for item in GRIPPER_TELEOP_SPECS)
    raise ValueError(f"Unsupported one-DoF gripper pair {robot}+{expected}; supported: {supported}")

def create_gripper_osc_pose_env(
    config: WristConfig,
    *,
    robot: str,
    gripper_type: str | None = None,
    has_renderer: bool,
    has_offscreen_renderer: bool = False,
):
    """Create the standard Lift cube/table environment for one official robot.

    ``load_composite_controller_config`` is queried with the actual robot name
    on every build.  The resulting OSC controller is explicitly configured as
    world-frame delta control using the already validated SpaceMouse physical
    limits.
    """

    spec = resolve_gripper_spec(robot, gripper_type)

    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import (
        load_composite_controller_config,
    )

    controller_config = copy.deepcopy(load_composite_controller_config(robot=robot))
    body_parts = controller_config.get("body_parts", {})
    if "right" not in body_parts:
        raise RuntimeError(f"{robot} controller configuration has no right-arm OSC section")
    arm_config = body_parts["right"]
    arm_config.update(
        input_type="delta",
        input_ref_frame="world",
        impedance_mode="fixed",
        output_min=[-config.max_translation_delta_m] * 3
        + [-config.max_rotation_delta_rad] * 3,
        output_max=[config.max_translation_delta_m] * 3
        + [config.max_rotation_delta_rad] * 3,
    )
    return suite.make(
        "Lift",
        robots=robot,
        gripper_types=spec.gripper_type,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=False,
        control_freq=int(round(config.control_hz)),
        horizon=10_000_000,
        ignore_done=True,
    )


def _finite_vector(value: Sequence[float], length: int) -> np.ndarray | None:
    """Return a finite float vector or ``None`` for a malformed command."""

    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


class OneDofGripperComposer:
    """Compose a world-frame OSC command and one installed gripper policy DoF.

    The physical gripper may expand this policy scalar into two Panda fingers
    or three Jaco fingers internally.  The policy slice is discovered from
    robosuite's composite-controller split table and must have exactly one
    entry.  This protects the Panda / Jaco entries from copying Allegro's
    sixteen-dimensional hand layout.
    """

    def __init__(self, env: Any):
        if len(getattr(env, "robots", ())) != 1:
            raise ValueError("one-DoF gripper teleoperation requires exactly one robot")
        self.env = env
        self.robot = env.robots[0]
        self._splits = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in dict(self.robot.composite_controller._action_split_indexes).items()
        }
        self._low, self._high = (
            np.asarray(value, dtype=np.float64) for value in env.action_spec
        )
        if (
            self._low.shape != self._high.shape
            or self._low.shape != (int(env.action_dim),)
            or not np.all(np.isfinite(self._low))
            or not np.all(np.isfinite(self._high))
            or not np.all(self._high >= self._low)
        ):
            raise RuntimeError("robosuite action_spec is incompatible with action_dim")
        if not np.all((self._low <= 0.0) & (0.0 <= self._high)):
            raise RuntimeError("this teleoperation path requires zero to be a valid hold action")

        arm_candidates = [
            name
            for name, controller in self.robot.part_controllers.items()
            if getattr(controller, "control_dim", None) == 6
            and controller.__class__.__name__ == "OperationalSpaceController"
            and name in self._splits
        ]
        if len(arm_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one six-dimensional OSC controller, found {arm_candidates}"
            )
        self.arm_part = arm_candidates[0]
        self.arm_slice = slice(*self._splits[self.arm_part])
        if self.arm_slice.stop - self.arm_slice.start != 6:
            raise RuntimeError("installed OSC action slice is not six-dimensional")
        self.controller = self.robot.part_controllers[self.arm_part]
        self.goal_guard = DesiredGoalGuard(self.controller)
        if (
            getattr(self.controller, "input_type", None) != "delta"
            or getattr(self.controller, "input_ref_frame", None) != "world"
        ):
            raise RuntimeError("OSC_POSE must be configured for world-frame delta control")

        policy_grippers = [
            (str(key), gripper)
            for key, gripper in getattr(self.robot, "gripper", {}).items()
            if int(getattr(gripper, "dof", -1)) == 1
        ]
        if len(policy_grippers) != 1:
            raise RuntimeError(
                "Expected exactly one official one-DoF gripper, found "
                f"{[(name, type(gripper).__name__) for name, gripper in policy_grippers]}"
            )
        self.gripper_key, self.gripper = policy_grippers[0]

        gripper_candidates = []
        for name, controller in self.robot.part_controllers.items():
            if name not in self._splits:
                continue
            action_slice = slice(*self._splits[name])
            if action_slice.stop - action_slice.start != 1:
                continue
            controller_name = controller.__class__.__name__.lower()
            # Prefer the installed controller that belongs to the discovered
            # gripper key.  The class-name fallback handles harmless naming
            # differences while still requiring exactly one one-DoF slice.
            belongs_to_gripper = name == f"{self.gripper_key}_gripper" or "grip" in controller_name
            if belongs_to_gripper:
                gripper_candidates.append((str(name), action_slice, controller))
        if len(gripper_candidates) != 1:
            raise RuntimeError(
                "Expected exactly one one-dimensional gripper policy slice, found "
                f"{[(name, [item.start, item.stop]) for name, item, _ in gripper_candidates]}"
            )
        self.gripper_part, self.gripper_slice, self.gripper_controller = gripper_candidates[0]
        self.gripper_low = float(self._low[self.gripper_slice][0])
        self.gripper_high = float(self._high[self.gripper_slice][0])
        if not self.gripper_low < self.gripper_high:
            raise RuntimeError("installed gripper action range is empty")

    def initialize_goal(self) -> None:
        """Latch the current end-effector pose to prevent a startup jump."""

        self.controller.reset_goal(goal_update_mode="desired")
        self.goal_guard.reset()

    def neutral_action(self) -> np.ndarray:
        """The robosuite no-op: zero OSC delta and zero / hold gripper action."""

        return np.zeros(int(self.env.action_dim), dtype=np.float64)

    def compose(
        self,
        physical_delta: Sequence[float],
        gripper_command: float = 0.0,
        *,
        base_action: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return a valid action that changes only the discovered two slices."""

        delta = _finite_vector(physical_delta, 6)
        if delta is None:
            delta = np.zeros(6, dtype=np.float64)
        if base_action is None:
            action = self.neutral_action()
        else:
            action = _finite_vector(base_action, int(self.env.action_dim))
            if action is None:
                raise ValueError(f"base_action must be a finite vector of length {self.env.action_dim}")
            action = action.copy()

        output_min = np.asarray(self.controller.output_min, dtype=np.float64)
        output_max = np.asarray(self.controller.output_max, dtype=np.float64)
        input_min = np.asarray(self.controller.input_min, dtype=np.float64)
        input_max = np.asarray(self.controller.input_max, dtype=np.float64)
        if (
            output_min.shape != (6,)
            or output_max.shape != (6,)
            or input_min.shape != (6,)
            or input_max.shape != (6,)
            or not np.all(output_max > output_min)
            or not np.all(input_max > input_min)
        ):
            raise RuntimeError("OSC_POSE has invalid action scaling limits")
        clipped_delta = np.clip(delta, output_min, output_max)
        self.goal_guard.before_step(clipped_delta)
        normalized = input_min + (clipped_delta - output_min) * (
            input_max - input_min
        ) / (output_max - output_min)
        action[self.arm_slice] = np.clip(
            normalized, self._low[self.arm_slice], self._high[self.arm_slice]
        )

        try:
            command = float(gripper_command)
        except (TypeError, ValueError, OverflowError):
            command = 0.0
        if not np.isfinite(command):
            command = 0.0
        action[self.gripper_slice] = np.clip(command, self.gripper_low, self.gripper_high)
        return action

    def describe(self) -> dict[str, Any]:
        """Return live API facts for logs and reproducible diagnostics."""

        gripper_joints = list(getattr(self.gripper, "joints", ()))
        return {
            "action_dim": int(self.env.action_dim),
            "action_spec": {"low": self._low.tolist(), "high": self._high.tolist()},
            "action_slices": {
                name: [int(start), int(stop)] for name, (start, stop) in self._splits.items()
            },
            "arm": {
                "part": self.arm_part,
                "slice": [self.arm_slice.start, self.arm_slice.stop],
                "controller": self.controller.__class__.__name__,
                "control_dim": int(getattr(self.controller, "control_dim", -1)),
                "input_type": getattr(self.controller, "input_type", None),
                "input_ref_frame": getattr(self.controller, "input_ref_frame", None),
                "physical_output_min": np.asarray(self.controller.output_min, dtype=np.float64).tolist(),
                "physical_output_max": np.asarray(self.controller.output_max, dtype=np.float64).tolist(),
            },
            "gripper": {
                "key": self.gripper_key,
                "part": self.gripper_part,
                "slice": [self.gripper_slice.start, self.gripper_slice.stop],
                "model": self.gripper.__class__.__name__,
                "policy_dof": int(getattr(self.gripper, "dof", -1)),
                "physical_controller_dim": int(getattr(self.gripper_controller, "control_dim", -1)),
                "joints": gripper_joints,
                "action_range": [self.gripper_low, self.gripper_high],
                "open_command": self.gripper_low,
                "close_command": self.gripper_high,
                "command_semantics": "official gripper convention: -1 open, +1 close",
            },
        }

    def ee_pose(self) -> dict[str, list[float]]:
        """Read the OSC controller's actual reference pose."""

        return {
            "position_m": np.asarray(self.controller.ref_pos, dtype=np.float64).tolist(),
            "rotation_matrix": np.asarray(self.controller.ref_ori_mat, dtype=np.float64)
            .reshape(-1)
            .tolist(),
        }


@dataclass(frozen=True)
class GripperCommand:
    """One gripper policy command plus the state that produced it."""

    value: float
    state: str
    rising_edges: tuple[str, ...]


class GripperButtonController:
    """Rising-edge SpaceMouse buttons for a generic one-DoF gripper.

    A left-button edge selects the upper action bound (close); a right-button
    edge selects the lower action bound (open).  Repeated reports while a
    button is held do not change state.  A stale or disconnected button packet
    immediately returns a zero / hold command and clears the selected state.
    """

    def __init__(self, config: WristConfig, *, open_command: float, close_command: float):
        if not np.isfinite(open_command) or not np.isfinite(close_command):
            raise ValueError("gripper action bounds must be finite")
        if not open_command < close_command:
            raise ValueError("open_command must be below close_command")
        self.left_button_index = int(config.hand_gestures.left_button_index)
        self.right_button_index = int(config.hand_gestures.right_button_index)
        self.open_command = float(open_command)
        self.close_command = float(close_command)
        self._previous: tuple[bool, ...] = ()
        self._command = 0.0
        self._state = "hold"

    @staticmethod
    def _button(buttons: tuple[bool, ...], index: int) -> bool:
        return 0 <= index < len(buttons) and bool(buttons[index])

    def reset(self) -> None:
        self._previous = ()
        self._command = 0.0
        self._state = "hold"

    def update(
        self,
        buttons: Sequence[bool],
        *,
        valid: bool,
        dt_s: float | None = None,
    ) -> GripperCommand:
        """Process a raw HID snapshot without exposing it to arm control."""

        # The generic Panda path is intentionally rising-edge based.  It
        # accepts the common loop API so the Jaco path can receive actual dt
        # without special-casing the safety-critical arm loop.
        del dt_s

        current = tuple(bool(value) for value in buttons)
        if not valid:
            self.reset()
            return GripperCommand(0.0, "safe_zero", ())

        left_edge = self._button(current, self.left_button_index) and not self._button(
            self._previous, self.left_button_index
        )
        right_edge = self._button(current, self.right_button_index) and not self._button(
            self._previous, self.right_button_index
        )
        # Simultaneous clicks are rare. Prefer open because it is the safer
        # physical outcome; either case still leaves the OSC arm unchanged.
        if right_edge:
            self._command = self.open_command
            self._state = "open"
        elif left_edge:
            self._command = self.close_command
            self._state = "close"
        self._previous = current
        edges = tuple(
            name
            for name, occurred in (("left_close", left_edge), ("right_open", right_edge))
            if occurred
        )
        return GripperCommand(self._command, self._state, edges)


class ThreeFingerContinuousGripperButtonController(GripperButtonController):
    """Level-triggered, rate-limited official three-finger gripper control.

    robosuite's installed :class:`SimpleGripController` interprets its one
    policy degree of freedom as a velocity, so sending a stale nonzero action
    after a button release would continue finger motion.  This controller
    keeps a bounded *policy target* for rate / limit accounting, but emits a
    velocity only while exactly one physical button is currently pressed:

    * left bit 0: close;
    * right bit 1: open;
    * neither / both: exact zero velocity and no target integration.

    The target advances by ``rate_per_s * actual_dt``.  It is clipped to the
    live gripper action limits, giving deterministic long-press behaviour and
    preventing any further accumulation after the policy limit is reached.
    A physical MuJoCo joint limit remains the final safety limit inside the
    official controller.
    """

    def __init__(
        self,
        config: WristConfig,
        *,
        open_command: float,
        close_command: float,
        rate_per_s: float = THREE_FINGER_GRIPPER_POLICY_TARGET_RATE_PER_S,
    ):
        super().__init__(config, open_command=open_command, close_command=close_command)
        if not np.isfinite(rate_per_s) or rate_per_s <= 0.0:
            raise ValueError("three-finger gripper rate_per_s must be finite and positive")
        self.rate_per_s = float(rate_per_s)
        self._target = 0.0
        self._last_direction = 0

    @property
    def target(self) -> float:
        """Bounded normalized policy target retained while the button is released."""

        return float(self._target)

    def reset(self) -> None:
        super().reset()
        self._target = 0.0
        self._last_direction = 0

    @staticmethod
    def _dt(value: float | None) -> float:
        if value is None:
            return 0.0
        try:
            dt = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return dt if np.isfinite(dt) and dt > 0.0 else 0.0

    def update(
        self,
        buttons: Sequence[bool],
        *,
        valid: bool,
        dt_s: float | None = None,
    ) -> GripperCommand:
        """Integrate one current button-level sample using the actual loop dt.

        ``value`` is the velocity input consumed by the official Jaco gripper
        controller.  It is zero on release, simultaneous button press, or
        stale input.  At a bounded policy target limit it remains in the
        requested direction so MuJoCo can finish travelling to its physical
        joint stop, while the policy target itself no longer accumulates.
        """

        current = tuple(bool(value) for value in buttons)
        if not valid:
            self.reset()
            return GripperCommand(0.0, "safe_zero", ())

        left_pressed = self._button(current, self.left_button_index)
        right_pressed = self._button(current, self.right_button_index)
        left_edge = left_pressed and not self._button(self._previous, self.left_button_index)
        right_edge = right_pressed and not self._button(self._previous, self.right_button_index)
        self._previous = current
        edges = tuple(
            name
            for name, occurred in (("left_close", left_edge), ("right_open", right_edge))
            if occurred
        )

        if left_pressed and right_pressed:
            self._state = "both_buttons_safe_hold"
            self._last_direction = 0
            return GripperCommand(0.0, self._state, edges)
        if not left_pressed and not right_pressed:
            # Hold the integrated target for telemetry / the next same-way
            # press, but deliberately send exact zero because SimpleGrip is
            # velocity-controlled.
            self._state = "hold"
            self._last_direction = 0
            return GripperCommand(0.0, self._state, edges)

        direction = 1 if left_pressed else -1
        # A direct reversal should never spend a visible period commanding the
        # previous physical direction.  Reset the velocity-target ramp before
        # integrating the newly requested direction.
        if self._target * direction < 0.0:
            self._target = 0.0
        previous_target = self._target
        increment = direction * self.rate_per_s * self._dt(dt_s)
        self._target = float(np.clip(
            previous_target + increment,
            self.open_command,
            self.close_command,
        ))
        self._last_direction = direction

        at_limit = bool(np.isclose(
            self._target,
            self.close_command if direction > 0 else self.open_command,
            atol=1.0e-12,
            rtol=0.0,
        ))
        self._state = (
            "close_limit" if direction > 0 else "open_limit"
        ) if at_limit else ("closing" if direction > 0 else "opening")

        # The target is a normalized velocity target.  It may be less than
        # the configured rate during a short press, and is clipped by the live
        # policy bounds so compose() cannot ever write an invalid action.
        command = self._target
        return GripperCommand(command, self._state, edges)

    def describe(self) -> dict[str, Any]:
        """Document the level-triggered semantics in metadata."""

        return {
            "mode": "level_triggered_continuous",
            "rate_policy_units_per_s": self.rate_per_s,
            "left_button_index": self.left_button_index,
            "left_button_action": "close_while_held",
            "right_button_index": self.right_button_index,
            "right_button_action": "open_while_held",
            "release_behavior": "zero gripper velocity; retain bounded policy target",
            "simultaneous_behavior": "zero gripper velocity; no target integration",
            "invalid_hid_behavior": "zero wrist delta and zero gripper velocity",
        }


# Backward-compatible name for local scripts and prior run logs. The same
# state machine is intentionally shared by Jaco and Robotiq three-finger
# grippers rather than being a Jaco-only special case.
JacoContinuousGripperButtonController = ThreeFingerContinuousGripperButtonController


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


class GripperRunLog:
    """Small self-contained telemetry writer for the two new entries."""

    def __init__(self, robot: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = ROOT / "outputs" / "gripper_teleop" / f"{stamp}_{robot.lower()}"
        self.path.mkdir(parents=True, exist_ok=False)
        self._stream = (self.path / "telemetry.jsonl").open("w", encoding="utf-8", newline="\n")

    def metadata(self, contents: Mapping[str, Any]) -> None:
        (self.path / "metadata.json").write_text(
            json.dumps(dict(contents), indent=2, default=_json_default), encoding="utf-8"
        )

    def event(self, contents: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(dict(contents), default=_json_default, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class _ViewerSmokeInput:
    """Bounded zero-motion input used only by ``--viewer-smoke``.

    It emits one synthetic close and one synthetic open edge so smoke tests
    verify the same button state machine without claiming a hardware test.
    """

    def __init__(self, config: WristConfig, *, continuous_buttons: bool):
        self.config = config
        self.continuous_buttons = bool(continuous_buttons)
        self._opened_at: float | None = None
        self._poll_count = 0
        self.closed = False

    def open(self) -> dict[str, Any]:
        self._opened_at = time.monotonic()
        self._poll_count = 0
        self.closed = False
        return {"source": "synthetic_viewer_smoke", "hardware": False}

    def reset_input_state(self) -> None:
        self._opened_at = time.monotonic()
        self._poll_count = 0

    def poll(self) -> MotionSample:
        now = time.monotonic()
        self._poll_count += 1
        # Use control-loop counts rather than wall time. Creating the native
        # MuJoCo viewer can briefly block the first simulation update, and a
        # wall-clock schedule could otherwise skip both smoke-test inputs.
        # Three-finger grippers deliberately receive two level-held ranges so
        # their smoke log
        # exercises continuous close, release / exact hold, and continuous
        # open. Panda retains its established edge-only pulse sequence.
        if self.continuous_buttons:
            close_pressed = 2 <= self._poll_count <= 14
            open_pressed = 19 <= self._poll_count <= 31
        else:
            close_pressed = self._poll_count == 2
            open_pressed = self._poll_count == 8
        return MotionSample(
            raw_axes=np.zeros(6, dtype=np.float64),
            buttons=(close_pressed, open_pressed),
            valid=True,
            armed=True,
            timestamp_s=now,
            buttons_valid=True,
        )

    def close(self) -> None:
        self.closed = True


class _OptionalVisualAssistance:
    """Keep optional shared visual aid separate from the control loop.

    A rendering problem is logged as telemetry and cannot prevent safe arm or
    gripper actions. When enabled, this wrapper owns the project's one laser
    and one synchronized compositor for Panda and Jaco alike.
    """

    def __init__(self, *, laser: Any | None, compositor: Any | None, reason: str | None):
        self.laser = laser
        self.compositor = compositor
        self.reason = reason

    @classmethod
    def create(
        cls,
        env: Any,
        config: WristConfig,
        *,
        enabled: bool,
        keypress_callback: Any,
    ) -> "_OptionalVisualAssistance":
        if not enabled:
            return cls(laser=None, compositor=None, reason="disabled_by_cli")
        tri_config = getattr(config, "tri_view", None)
        laser_config = getattr(config, "laser", None)
        if tri_config is None and laser_config is None:
            return cls(laser=None, compositor=None, reason="visualization_config_not_available")
        try:
            from .visualization import LaserPointer, TriViewCompositor
        except ImportError as exc:
            return cls(laser=None, compositor=None, reason=f"visualization_import_unavailable: {exc}")
        try:
            laser = LaserPointer(env, config=laser_config) if laser_config is not None else None
            compositor = (
                TriViewCompositor(
                    tri_config,
                    key_handler=keypress_callback,
                    on_close=lambda: keypress_callback(ord("q")),
                    laser=laser,
                )
                if tri_config is not None
                else None
            )
            return cls(laser=laser, compositor=compositor, reason=None)
        except Exception as exc:  # Optional display must never stop arm safety.
            return cls(
                laser=None,
                compositor=None,
                reason=f"visualization_initialization_failed: {type(exc).__name__}: {exc}",
            )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.reason is None,
            "laser": self.laser is not None,
            "tri_view": self.compositor is not None,
            "reason": self.reason,
        }

    def configure_control_frame(self, env: Any, mapper: SixDofMapper) -> dict[str, Any]:
        """Apply the shared primary camera and bind its live basis to input."""
        if self.compositor is None:
            return mapper.describe_control_frame()
        self.compositor.configure_cameras(env)
        return mapper.set_camera_from_sim(env, self.compositor.config.front_camera_name)

    def update(self, env: Any, *, status_text: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            if self.laser is not None:
                laser_record = self.laser.update()
                if isinstance(laser_record, Mapping):
                    result["laser"] = dict(laser_record)
            if self.compositor is not None:
                maybe_result = self.compositor.update(env, status_text=status_text)
                if isinstance(maybe_result, Mapping):
                    result.update(maybe_result)
        except Exception as exc:
            # Visualization remains explicitly non-physical and cannot be
            # allowed to interrupt the safety-critical action loop.
            result["visualization_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def save_last_frame(self, path: Path) -> bool:
        """Persist one already-rendered composite as non-physical evidence."""
        writer = getattr(self.compositor, "save_last_frame", None)
        if not callable(writer):
            return False
        try:
            return bool(writer(path))
        except Exception:
            return False

    def close(self) -> None:
        for item in (self.compositor, self.laser):
            close = getattr(item, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


_FAST_JACO_CLASSES: dict[type, type] = {}


def apply_jaco_gripper_speed(env: Any, speed_per_step: float = JACO_GRIPPER_TARGET_SPEED_PER_STEP) -> int:
    """Give every live JacoThreeFingerGripper a faster, magnitude-aware sweep.

    The installed class is swapped for a subclass on this instance only; the
    installed package is never edited. Call after every ``env.reset()``
    because robosuite's hard reset rebuilds the gripper objects.
    """

    if not np.isfinite(speed_per_step) or not 0.0 < float(speed_per_step) <= 1.0:
        raise ValueError("Jaco gripper speed must be in (0, 1] per policy step")
    patched = 0
    for robot in getattr(env, "robots", ()):
        grippers = getattr(robot, "gripper", {})
        for gripper in (grippers.values() if isinstance(grippers, dict) else (grippers,)):
            base = type(gripper)
            if getattr(base, "_project_fast_jaco", False):
                base = base.__mro__[1]
            elif base.__name__ != "JacoThreeFingerGripper":
                continue
            fast = _FAST_JACO_CLASSES.get(base)
            if fast is None:
                def format_action(self: Any, action: Any) -> np.ndarray:
                    assert len(action) == self.dof
                    command = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
                    # Same sign convention as the installed class.
                    self.current_action = np.clip(self.current_action - self.speed * command, -1.0, 1.0)
                    return self.current_action * np.array([1, 1, 1])

                fast = type(base.__name__, (base,), {  # keep the name entry checks expect
                    "_project_fast_jaco": True,
                    "format_action": format_action,
                    "speed": property(lambda self: self._project_speed),
                })
                _FAST_JACO_CLASSES[base] = fast
            gripper.__class__ = fast
            gripper._project_speed = float(speed_per_step)
            patched += 1
    return patched


def reset_gripper_environment(
    env: Any,
    *,
    mapper: SixDofMapper,
    device: Any,
    buttons: GripperButtonController,
    keypress_callback: Any | None,
) -> tuple[OneDofGripperComposer, np.ndarray]:
    """Implement one complete R-key reset while retaining the HID handle."""

    env.reset()
    apply_jaco_gripper_speed(env)
    mapper.reset()
    reset_input = getattr(device, "reset_input_state", None)
    if callable(reset_input):
        reset_input()
    buttons.reset()
    composer = OneDofGripperComposer(env)
    composer.initialize_goal()
    if keypress_callback is not None:
        install_viewer_key_callback(env, keypress_callback)
    initial_action = composer.compose(np.zeros(6, dtype=np.float64), 0.0)
    env.step(initial_action)
    return composer, initial_action


def close_gripper_resources(
    *,
    env: Any | None,
    composer: OneDofGripperComposer | None,
    mapper: SixDofMapper,
    device: Any,
    visual: _OptionalVisualAssistance | None,
) -> np.ndarray | None:
    """Send one final zero action, then release every owned resource."""

    last_action = None
    try:
        if env is not None and composer is not None:
            last_action = composer.compose(np.zeros(6, dtype=np.float64), 0.0)
            env.step(last_action)
    finally:
        mapper.reset()
        if visual is not None:
            visual.close()
        close = getattr(device, "close", None)
        if callable(close):
            close()
        if env is not None:
            env.close()
    return last_action


def _sample_status(sample: MotionSample, command: np.ndarray, mapper: SixDofMapper) -> dict[str, Any]:
    return {
        "raw_axes": sample.raw_axes,
        "calibrated_axes": sample.control_axes,
        "zero_bias_axes": sample.zero_bias_axes,
        "neutral_locked": sample.neutral_locked,
        "input_calibration_state": sample.calibration_state,
        "mapped_world_delta": command,
        "mapper": mapper.diagnostics(),
        "motion_valid": sample.valid,
        "armed": sample.armed,
        "buttons": sample.buttons,
        "buttons_valid": sample.buttons_valid,
        "device_error": sample.error,
    }


def _sleep_to_rate(cycle_start: float, control_hz: float) -> None:
    remaining = 1.0 / control_hz - (time.monotonic() - cycle_start)
    if remaining > 0.0:
        time.sleep(remaining)


def run_gripper_teleop(
    *,
    robot: str,
    gripper_type: str | None = None,
    config: WristConfig,
    headless: bool,
    duration_s: float | None,
    viewer_smoke: bool = False,
    headless_smoke: bool = False,
    enable_three_view: bool = True,
    cube_enlarge_factor: float = 1.5,
    record_path: Path | None = None,
    three_finger_rate_per_s: float | None = None,
) -> int:
    """Run the shared loop for the project's tested one-DoF configurations.

    Normal runs require a real SpaceMouse.  ``viewer_smoke`` intentionally
    substitutes bounded synthetic zero motion plus one close/open button
    sequence; its metadata marks that path as non-hardware validation.
    """

    if viewer_smoke and headless:
        raise ValueError("--viewer-smoke requires a visible MuJoCo viewer")
    if headless_smoke and not headless:
        raise ValueError("--headless-smoke requires headless=True")
    if viewer_smoke and headless_smoke:
        raise ValueError("choose either viewer_smoke or headless_smoke")
    if duration_s is not None and duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if not np.isfinite(cube_enlarge_factor) or cube_enlarge_factor <= 1.0:
        raise ValueError("cube_enlarge_factor must be finite and greater than one")

    spec = resolve_gripper_spec(robot, gripper_type)
    if three_finger_rate_per_s is not None:
        if not spec.continuous_buttons:
            raise ValueError("three_finger_rate_per_s applies only to continuous three-finger grippers")
        if not np.isfinite(three_finger_rate_per_s) or three_finger_rate_per_s <= 0.0:
            raise ValueError("three_finger_rate_per_s must be finite and positive")
    active_three_finger_rate = (
        float(three_finger_rate_per_s)
        if three_finger_rate_per_s is not None
        else spec.continuous_rate_per_s
    )
    run_label = f"{spec.robot}_{spec.gripper_type}".replace("Gripper", "").lower()
    log = GripperRunLog(run_label)
    mapper = SixDofMapper(config)
    synthetic_smoke = bool(viewer_smoke or headless_smoke)
    device: Any = (
        _ViewerSmokeInput(config, continuous_buttons=spec.continuous_buttons)
        if synthetic_smoke
        else SpaceMouseHid(config)
    )
    keyboard = KeyboardCommandRouter()
    env = None
    composer: OneDofGripperComposer | None = None
    buttons: GripperButtonController | None = None
    visual: _OptionalVisualAssistance | None = None
    cube: InteractiveCubeScaler | None = None
    recorder = SimVideoRecorder(record_path, fps=config.control_hz)
    visual_snapshot: Path | None = None
    grasp: GraspTelemetry | None = None
    control_frame: dict[str, Any] = mapper.describe_control_frame()
    reason = "completed"
    exit_code = 0
    cube_resize_requested = False
    if synthetic_smoke and duration_s is None:
        duration_s = 2.5

    def route_viewer_key(key: int) -> None:
        nonlocal cube_resize_requested
        keyboard.feed(key, source="mujoco")
        if isinstance(key, int) and 0 <= key <= 255 and chr(key).lower() == "b":
            cube_resize_requested = True

    try:
        wants_tri_view = bool(
            not headless
            and enable_three_view
            and getattr(getattr(config, "tri_view", None), "enabled", False)
        )
        env = create_gripper_osc_pose_env(
            config,
            robot=spec.robot,
            gripper_type=spec.gripper_type,
            # The compositor owns the only visible window when enabled.
            # Avoid creating a competing native MuJoCo viewer in that case.
            has_renderer=not headless and not wants_tri_view,
            has_offscreen_renderer=wants_tri_view or record_path is not None,
        )
        device_info = device.open()
        # Build a temporary composer only to obtain verified live action
        # bounds for the button state machine; reset builds the final one.
        env.reset()
        probe = OneDofGripperComposer(env)
        buttons = (
            ThreeFingerContinuousGripperButtonController(
                config,
                open_command=probe.gripper_low,
                close_command=probe.gripper_high,
                rate_per_s=float(active_three_finger_rate),
            )
            if spec.continuous_buttons
            else GripperButtonController(
                config,
                open_command=probe.gripper_low,
                close_command=probe.gripper_high,
            )
        )
        composer, initial_action = reset_gripper_environment(
            env,
            mapper=mapper,
            device=device,
            buttons=buttons,
            keypress_callback=None if headless or wants_tri_view else route_viewer_key,
        )
        expected_gripper = spec.gripper_type
        if composer.gripper.__class__.__name__ != expected_gripper:
            raise RuntimeError(
                f"{spec.robot} loaded {composer.gripper.__class__.__name__}, expected {expected_gripper}"
            )
        cube = InteractiveCubeScaler(env, factor=cube_enlarge_factor)
        visual = _OptionalVisualAssistance.create(
            env,
            config,
            enabled=not headless and enable_three_view,
            keypress_callback=route_viewer_key,
        )
        control_frame = visual.configure_control_frame(env, mapper)
        grasp = GraspTelemetry(env, config.grasp_telemetry)
        log.metadata(
            {
                "mode": "one_dof_gripper_teleop",
                "robot": spec.robot,
                "gripper_type": spec.gripper_type,
                "finger_count": spec.fingers,
                "headless": headless,
                "viewer_smoke": viewer_smoke,
                "headless_smoke": headless_smoke,
                "config": config.to_dict(),
                "device": device_info,
                "osc_pose_and_gripper": composer.describe(),
                "control_frame": control_frame,
                "button_semantics": (
                    buttons.describe()
                    if isinstance(buttons, ThreeFingerContinuousGripperButtonController)
                    else {
                        "mode": "rising_edge_selected_command",
                        "left_button_index": buttons.left_button_index,
                        "left_button_action": "close",
                        "right_button_index": buttons.right_button_index,
                        "right_button_action": "open",
                        "rising_edge_only": True,
                        "invalid_hid_behavior": "zero wrist delta and zero gripper hold command",
                    }
                ),
                "three_finger_rate_override_policy_units_per_s": three_finger_rate_per_s,
                "keyboard": {
                    "reset": "R/r resets environment, OSC goal, mapper, button state",
                    "quit": "Q/q and Ctrl+C send a final zero action then release resources",
                },
                "visualization": visual.describe(),
                "interactive_cube": cube.status(),
                "recording": recorder.summary(),
                "grasp_telemetry": {
                    "config": config.grasp_telemetry.to_dict(),
                    "initial": grasp.update(),
                },
            }
        )
        print(
            f"{spec.robot} + {composer.gripper.__class__.__name__} ready: "
            "SpaceMouse input uses the primary-camera frame and drives world-frame OSC_POSE; "
            "left closes and right opens the gripper."
        )
        if isinstance(buttons, ThreeFingerContinuousGripperButtonController):
            print(
                "Three-finger gripper button mapping: "
                f"left[{buttons.left_button_index}]=close while held, "
                f"right[{buttons.right_button_index}]=open while held, "
                f"rate={buttons.rate_per_s:.3g} policy-units/s."
            )
        print("B enlarges the Lift cube once. R resets. Q or Ctrl+C exits safely.")
        log.event(
            {
                "event": "environment_reset",
                "reason": "startup",
                "timestamp_monotonic_s": time.monotonic(),
                "action": initial_action,
            }
        )

        start = previous = last_status = time.monotonic()
        while duration_s is None or time.monotonic() - start < duration_s:
            if keyboard.stop_event.is_set():
                reason = keyboard.quit_reason or "Q"
                break
            if keyboard.take_reset():
                # env.reset() reconstructs MuJoCo's state. Rebind the visual
                # site / compositor objects afterward so their handles never
                # point at a pre-reset scene.
                if visual is not None:
                    visual.close()
                if cube is not None and cube.enlarged:
                    log.event({
                        "event": "cube_size_restored_before_reset",
                        "timestamp_monotonic_s": time.monotonic(),
                        "cube": cube.restore_default(),
                    })
                composer, initial_action = reset_gripper_environment(
                    env,
                    mapper=mapper,
                    device=device,
                    buttons=buttons,
                    keypress_callback=None if headless or wants_tri_view else route_viewer_key,
                )
                visual = _OptionalVisualAssistance.create(
                    env,
                    config,
                    enabled=not headless and enable_three_view,
                    keypress_callback=route_viewer_key,
                )
                control_frame = visual.configure_control_frame(env, mapper)
                grasp = GraspTelemetry(env, config.grasp_telemetry)
                cube = InteractiveCubeScaler(env, factor=cube_enlarge_factor)
                log.event(
                    {
                        "event": "environment_reset",
                        "reason": "R",
                        "timestamp_monotonic_s": time.monotonic(),
                        "action": initial_action,
                        "visualization": visual.describe(),
                        "control_frame": control_frame,
                        "grasp": grasp.update(),
                    }
                )
                print("environment_reset")
                previous = time.monotonic()
                continue

            if cube_resize_requested:
                cube_resize_requested = False
                cube_event = cube.enlarge() if cube is not None else {
                    "changed": False,
                    "reason": "cube_scaler_not_initialized",
                }
                log.event({
                    "event": "cube_size_request",
                    "timestamp_monotonic_s": time.monotonic(),
                    "cube": cube_event,
                })
                print(f"cube resize: {cube_event}")

            cycle_start = time.monotonic()
            # Measured control-loop duration, not a fixed nominal period.
            # It becomes the Jaco target integration interval and remains
            # telemetry for both robots.
            elapsed = max(cycle_start - previous, 0.0)
            sample = device.poll()
            wrist_delta = mapper.map(
                sample.control_axes,
                valid=bool(sample.valid and sample.armed),
            )
            gripper = buttons.update(
                sample.buttons,
                valid=bool(sample.buttons_valid and sample.armed),
                dt_s=elapsed,
            )
            action = composer.compose(wrist_delta, gripper.value)
            _observation, reward, done, _info = env.step(action)
            recorder.capture(env)
            grasp_event = grasp.update() if grasp is not None else {"available": False, "reason": "not_initialized"}
            previous = cycle_start
            visual_event = visual.update(
                env,
                status_text=(
                    f"{spec.robot}+{spec.gripper_type} | gripper={gripper.state} | "
                    f"buttons={','.join(gripper.rising_edges) or '-'}"
                ),
            ) if visual is not None else {}
            if (
                visual is not None
                and visual_snapshot is None
                and bool(visual_event.get("rendered", False))
            ):
                candidate = log.path / "tri_view_laser_initial.png"
                if visual.save_last_frame(candidate):
                    visual_snapshot = candidate
                    visual_event["screenshot"] = str(candidate)
                else:
                    visual_event["screenshot_error"] = "could_not_write_tri_view_frame"
            if bool(visual_event.get("closed", False)):
                keyboard.feed("q", source="tri_view")

            event = _sample_status(sample, wrist_delta, mapper)
            event.update(
                {
                    "event": "step",
                    "timestamp_monotonic_s": cycle_start,
                    # ``elapsed`` is allowed to be exactly zero on the first
                    # high-resolution timer tick. Do not invent integration
                    # time for Jaco, but keep the diagnostic rate finite.
                    "update_hz": 1.0 / max(elapsed, 1.0e-9),
                    "action": action,
                    "arm_action": action[composer.arm_slice],
                    "gripper_action": action[composer.gripper_slice],
                    "gripper_state": gripper.state,
                    "button_rising_edges": gripper.rising_edges,
                    "button_levels": tuple(bool(value) for value in sample.buttons),
                    "gripper_policy_target": getattr(buttons, "target", None),
                    "gripper_rate_policy_units_per_s": getattr(buttons, "rate_per_s", None),
                    "gripper_integration_dt_s": elapsed,
                    "ee_pose": composer.ee_pose(),
                    "reward": reward,
                    "done": done,
                    "visualization": visual_event,
                    "recording": recorder.summary() if recorder.frame_count == 1 else None,
                    "interactive_cube": cube.status() if cube is not None else None,
                    "grasp": grasp_event,
                    "loop_seconds": time.monotonic() - cycle_start,
                }
            )
            log.event(event)
            if cycle_start - last_status >= 1.0:
                print(
                    f"{spec.robot}+{spec.gripper_type} step: wrist={np.round(wrist_delta, 5).tolist()} "
                    f"gripper={gripper.state} update={event['update_hz']:.1f}Hz"
                )
                last_status = cycle_start
            _sleep_to_rate(cycle_start, config.control_hz)
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    except Exception as exc:
        reason = f"error: {type(exc).__name__}: {exc}"
        exit_code = 1
        log.event({"event": "exception", "reason": reason, "timestamp_monotonic_s": time.monotonic()})
        print(reason, file=sys.stderr)
    finally:
        last_action = None
        try:
            if grasp is not None:
                log.event(
                    {
                        "event": "grasp_summary",
                        "timestamp_monotonic_s": time.monotonic(),
                        "grasp": grasp.summary(),
                    }
                )
            recorder.close()
            log.event({
                "event": "recording_summary",
                "timestamp_monotonic_s": time.monotonic(),
                "recording": recorder.summary(),
            })
            last_action = close_gripper_resources(
                env=env,
                composer=composer,
                mapper=mapper,
                device=device,
                visual=visual,
            )
        except Exception as exc:
            cleanup_reason = f"cleanup error: {type(exc).__name__}: {exc}"
            log.event(
                {
                    "event": "cleanup_exception",
                    "reason": cleanup_reason,
                    "timestamp_monotonic_s": time.monotonic(),
                }
            )
            print(cleanup_reason, file=sys.stderr)
            if exit_code == 0:
                exit_code = 1
                reason = cleanup_reason
        log.event(
            {
                "event": "exit",
                "reason": reason,
                "timestamp_monotonic_s": time.monotonic(),
                "last_zero_action": last_action,
            }
        )
        log.close()
        print(f"{spec.robot}+{spec.gripper_type} teleoperation stopped: {reason}. Log: {log.path}")
    return exit_code
