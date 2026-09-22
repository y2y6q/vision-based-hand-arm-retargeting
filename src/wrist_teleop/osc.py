"""Verified robosuite OSC_POSE setup and wrist-only action composition."""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np

from .config import WristConfig


def create_panda_osc_pose_env(config: WristConfig, *, has_renderer: bool):
    """Create the sole Panda wrist teleoperation environment.

    The installed robosuite 1.5.2 controller consumes a normalized six-vector;
    it internally scales this to physical deltas.  The controller output limits
    are set from the same validated config that clamps the SpaceMouse mapper.
    This makes the interface between mapper and composer physical metres /
    radians while preserving robosuite's actual OSC_POSE API.
    """
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import (
        load_composite_controller_config,
    )

    controller_config = copy.deepcopy(load_composite_controller_config(robot="Panda"))
    arm_config = controller_config["body_parts"]["right"]
    arm_config.update(
        input_type="delta",
        input_ref_frame="world",
        impedance_mode="fixed",
        output_min=[-config.max_translation_delta_m] * 3 + [-config.max_rotation_delta_rad] * 3,
        output_max=[config.max_translation_delta_m] * 3 + [config.max_rotation_delta_rad] * 3,
    )
    return suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=int(round(config.control_hz)),
        horizon=10_000_000,
        ignore_done=True,
    )


class OscPoseComposer:
    """Maps a physical world delta into the installed OSC_POSE arm slice only."""

    def __init__(self, env: Any):
        if len(env.robots) != 1:
            raise ValueError("wrist teleoperation requires exactly one Panda robot")
        self.env = env
        self.robot = env.robots[0]
        self._splits = dict(self.robot.composite_controller._action_split_indexes)
        candidates = [
            name for name, controller in self.robot.part_controllers.items()
            if getattr(controller, "control_dim", None) == 6
            and controller.__class__.__name__ == "OperationalSpaceController"
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one six-dimensional OSC arm controller, found {candidates}")
        self.arm_name = candidates[0]
        if self.arm_name not in self._splits:
            raise RuntimeError(f"OSC controller {self.arm_name!r} has no action slice")
        self.arm_slice = slice(*self._splits[self.arm_name])
        if self.arm_slice.stop - self.arm_slice.start != 6:
            raise RuntimeError("installed OSC_POSE action slice is not six-dimensional")
        self.controller = self.robot.part_controllers[self.arm_name]
        if self.controller.input_type != "delta" or self.controller.input_ref_frame != "world":
            raise RuntimeError("OSC_POSE must be configured as world-frame delta control")
        self._low, self._high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
        if self._low.shape != self._high.shape or self._low.size != env.action_dim:
            raise RuntimeError("robosuite action_spec disagrees with action_dim")

    def initialize_goal(self) -> None:
        """Latch current EE pose as desired goal, preventing startup jumps."""
        self.controller.reset_goal(goal_update_mode="desired")

    def neutral_action(self) -> np.ndarray:
        """Robosuite's verified no-op: zero OSC delta and zero gripper action."""
        return np.zeros(self.env.action_dim, dtype=np.float64)

    def compose(self, physical_delta: Sequence[float], base_action: Sequence[float] | None = None) -> np.ndarray:
        """Return a valid action while changing only the six OSC arm entries.

        ``physical_delta`` is ``[dx, dy, dz, rx, ry, rz]`` in world metres /
        axis-angle radians. ``base_action`` is copied unchanged outside the
        arm slice. The wrist-only entry leaves it absent; integrated mode may
        separately provide bounded hand targets, but this OSC composer never
        derives those targets from motion axes or button state.
        """
        delta = np.asarray(physical_delta, dtype=np.float64)
        if delta.shape != (6,) or not np.all(np.isfinite(delta)):
            delta = np.zeros(6, dtype=np.float64)
        if base_action is None:
            action = self.neutral_action()
        else:
            action = np.asarray(base_action, dtype=np.float64).copy()
            if action.shape != (self.env.action_dim,) or not np.all(np.isfinite(action)):
                raise ValueError(f"base_action must be a finite vector of length {self.env.action_dim}")

        output_min = np.asarray(self.controller.output_min, dtype=np.float64)
        output_max = np.asarray(self.controller.output_max, dtype=np.float64)
        input_min = np.asarray(self.controller.input_min, dtype=np.float64)
        input_max = np.asarray(self.controller.input_max, dtype=np.float64)
        if not np.all(output_max > output_min) or not np.all(input_max > input_min):
            raise RuntimeError("OSC_POSE has invalid action scaling limits")
        clipped_delta = np.clip(delta, output_min, output_max)
        normalized = input_min + (clipped_delta - output_min) * (input_max - input_min) / (output_max - output_min)
        action[self.arm_slice] = np.clip(normalized, self._low[self.arm_slice], self._high[self.arm_slice])
        return action

    def describe(self) -> dict[str, Any]:
        """JSON-ready installed-API facts for logs and reproducibility."""
        return {
            "action_dim": int(self.env.action_dim),
            "action_spec": {"low": self._low.tolist(), "high": self._high.tolist()},
            "action_slices": {name: [int(start), int(stop)] for name, (start, stop) in self._splits.items()},
            "arm_part": self.arm_name,
            "arm_slice": [self.arm_slice.start, self.arm_slice.stop],
            "controller": "OSC_POSE",
            "input_type": self.controller.input_type,
            "input_ref_frame": self.controller.input_ref_frame,
            "physical_output_min": np.asarray(self.controller.output_min).tolist(),
            "physical_output_max": np.asarray(self.controller.output_max).tolist(),
        }

    def ee_pose(self) -> dict[str, list[float]]:
        # ``robot._hand_pose`` is a robot-model transform, not necessarily the
        # OSC reference site. The controller's reference is the EE pose that
        # OSC actually regulates and is refreshed in every ``set_goal`` call.
        return {
            "position_m": np.asarray(self.controller.ref_pos, dtype=np.float64).tolist(),
            "rotation_matrix": np.asarray(self.controller.ref_ori_mat, dtype=np.float64).reshape(-1).tolist(),
        }
