"""Independent Panda OSC and 20-actuator Shadow Hand action composition."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .osc import OscPoseComposer
from .shadow_gripper import SHADOW_ACTUATOR_NAMES, SHADOW_JOINT_NAMES


class PandaShadowActionComposer(OscPoseComposer):
    """Compose Panda's six OSC actions and Shadow Hand's 20 actuator targets.

    The hand receives official position-actuator targets, in the fixed order
    declared by :data:`SHADOW_ACTUATOR_NAMES`.  It never accepts a 24-vector:
    Shadow's four distal pairs are controlled by four tendon actuators.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        candidates = [g for g in self.robot.gripper.values() if int(getattr(g, "dof", -1)) == 20]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one 20-actuator Shadow gripper, found {len(candidates)}")
        self.hand = candidates[0]
        required = ("actuator_names", "joint_names", "lower_limits", "upper_limits", "targets_to_action", "action_to_targets")
        missing = [name for name in required if not hasattr(self.hand, name)]
        if missing:
            raise RuntimeError(f"Shadow gripper lacks integration API: {missing}")
        if tuple(self.hand.actuator_names) != SHADOW_ACTUATOR_NAMES or tuple(self.hand.joint_names) != SHADOW_JOINT_NAMES:
            raise RuntimeError("Shadow model action or joint contract diverged from the verified official source")
        self.hand_lower_limits = np.asarray(self.hand.lower_limits, dtype=np.float64)
        self.hand_upper_limits = np.asarray(self.hand.upper_limits, dtype=np.float64)
        if self.hand_lower_limits.shape != (20,) or self.hand_upper_limits.shape != (20,):
            raise RuntimeError("Shadow actuator limits must be 20-dimensional")
        if not np.all(np.isfinite(self.hand_lower_limits)) or not np.all(np.isfinite(self.hand_upper_limits)) or not np.all(self.hand_upper_limits > self.hand_lower_limits):
            raise RuntimeError("Shadow actuator limits are invalid")
        parts = [name for name, controller in self.robot.part_controllers.items() if name != self.arm_name and int(getattr(controller, "control_dim", -1)) == 20]
        if len(parts) != 1 or parts[0] not in self._splits:
            raise RuntimeError(f"Expected one 20-dimensional Shadow controller, found {parts}")
        self.hand_name = parts[0]
        self.hand_slice = slice(*self._splits[self.hand_name])
        if self.hand_slice.stop - self.hand_slice.start != 20:
            raise RuntimeError("Shadow action slice is not 20-dimensional")
        hand_key = next(key for key, gripper in self.robot.gripper.items() if gripper is self.hand)
        self._hand_qpos_indexes = np.asarray(self.robot._ref_gripper_joint_pos_indexes[hand_key], dtype=np.intp)
        if self._hand_qpos_indexes.shape != (24,):
            raise RuntimeError("Shadow qpos contract must expose 24 physical joints")
        self._held_targets = np.asarray(self.hand.action_to_targets(self.hand.initial_action), dtype=np.float64)
        self._validate_mapping()

    def _validate_mapping(self) -> None:
        lower_action = np.asarray(self.hand.targets_to_action(self.hand_lower_limits), dtype=np.float64)
        upper_action = np.asarray(self.hand.targets_to_action(self.hand_upper_limits), dtype=np.float64)
        if lower_action.shape != (20,) or upper_action.shape != (20,):
            raise RuntimeError("Shadow target mapping returned the wrong dimension")
        if not np.allclose(lower_action, self._low[self.hand_slice]) or not np.allclose(upper_action, self._high[self.hand_slice]):
            raise RuntimeError("Shadow target mapping does not match the live action specification")
        if not np.allclose(self.hand.action_to_targets(lower_action), self.hand_lower_limits, atol=1e-8, rtol=0.0):
            raise RuntimeError("Shadow lower action mapping does not round-trip")

    def hand_qpos(self) -> np.ndarray:
        """Return the 24 physical joint positions in official MJCF joint order."""
        return np.asarray(self.env.sim.data.qpos[self._hand_qpos_indexes], dtype=np.float64).copy()

    @property
    def held_hand_targets(self) -> np.ndarray:
        return self._held_targets.copy()

    def hand_action_for_targets(self, targets: Sequence[float] | None) -> np.ndarray:
        if targets is not None:
            values = np.asarray(targets, dtype=np.float64)
            if values.shape == (20,) and np.all(np.isfinite(values)):
                self._held_targets = np.clip(values, self.hand_lower_limits, self.hand_upper_limits)
        normalized = np.asarray(self.hand.targets_to_action(self._held_targets), dtype=np.float64)
        return np.clip(normalized, self._low[self.hand_slice], self._high[self.hand_slice])

    def hold_action(self, base_action: Sequence[float] | None = None) -> np.ndarray:
        if base_action is None:
            action = self.neutral_action()
        else:
            action = np.asarray(base_action, dtype=np.float64).copy()
            if action.shape != (self.env.action_dim,) or not np.all(np.isfinite(action)):
                raise ValueError(f"base_action must be a finite vector of length {self.env.action_dim}")
        action[self.hand_slice] = self.hand_action_for_targets(None)
        return action

    def compose(self, physical_delta: Sequence[float], hand_targets: Sequence[float] | None = None, base_action: Sequence[float] | None = None) -> np.ndarray:
        action = self.hold_action(base_action)
        action[self.hand_slice] = self.hand_action_for_targets(hand_targets)
        return super().compose(physical_delta, action)

    def describe(self) -> dict[str, Any]:
        details = super().describe()
        details.update(
            hand_part=self.hand_name,
            hand_slice=[self.hand_slice.start, self.hand_slice.stop],
            physical_hand_joint_count=24,
            hand_actuator_count=20,
            hand_joint_names=list(SHADOW_JOINT_NAMES),
            hand_actuator_names=list(SHADOW_ACTUATOR_NAMES),
            tendon_coupled_actuators=["rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"],
            hand_tcp_site=self.hand.important_sites["grip_site"],
        )
        return details
