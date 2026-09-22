"""Independent Panda OSC wrist and Allegro position-action composition.

The module is deliberately small: :class:`PandaAllegroActionComposer` owns
the two verified robosuite action slices and never accepts a camera wrist
command or a SpaceMouse hand command.  The Allegro gripper uses MuJoCo
position actuators, so camera targets are absolute joint radians; they are
converted through the custom gripper's installed ``[-1, 1]`` action mapping.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .osc import OscPoseComposer


class PandaAllegroActionComposer(OscPoseComposer):
    """Compose the Panda's six OSC entries and the Allegro's sixteen entries.

    ``compose(wrist_delta, hand_targets)`` is the only integration boundary.
    ``wrist_delta`` is a world-frame ``[Tx, Ty, Tz, Rx, Ry, Rz]`` physical
    delta handled by :class:`OscPoseComposer`. ``hand_targets`` is in the
    exact source Allegro joint order exposed by the installed custom gripper.
    Supplying ``None`` (or an invalid target) retains the last safe target.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        grippers = [gripper for gripper in self.robot.gripper.values() if int(getattr(gripper, "dof", -1)) == 16]
        if len(grippers) != 1:
            raise RuntimeError(f"Expected exactly one 16-DoF Allegro gripper, found {len(grippers)}")
        self.hand = grippers[0]

        candidates = [
            name
            for name, controller in self.robot.part_controllers.items()
            if name != self.arm_name and int(getattr(controller, "control_dim", -1)) == 16
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one 16-dimensional hand controller, found {candidates}")
        self.hand_name = candidates[0]
        if self.hand_name not in self._splits:
            raise RuntimeError(f"Allegro controller {self.hand_name!r} has no action slice")
        self.hand_slice = slice(*self._splits[self.hand_name])
        if self.hand_slice.stop - self.hand_slice.start != 16:
            raise RuntimeError("Allegro action slice is not sixteen-dimensional")

        required = ("joint_names", "lower_limits", "upper_limits", "joint_targets_to_action", "action_to_joint_targets")
        missing = [name for name in required if not hasattr(self.hand, name)]
        if missing:
            raise RuntimeError(f"Allegro gripper lacks required integration API: {missing}")
        self.hand_joint_names = tuple(self.hand.joint_names)
        self.hand_lower_limits = np.asarray(self.hand.lower_limits, dtype=np.float64)
        self.hand_upper_limits = np.asarray(self.hand.upper_limits, dtype=np.float64)
        if (len(self.hand_joint_names) != 16 or self.hand_lower_limits.shape != (16,)
                or self.hand_upper_limits.shape != (16,) or len(set(self.hand_joint_names)) != 16
                or not np.all(np.isfinite(self.hand_lower_limits))
                or not np.all(np.isfinite(self.hand_upper_limits))
                or not np.all(self.hand_upper_limits > self.hand_lower_limits)):
            raise RuntimeError("Allegro model has invalid joint names or limits")

        hand_key = next(key for key, gripper in self.robot.gripper.items() if gripper is self.hand)
        qpos_indexes = np.asarray(self.robot._ref_gripper_joint_pos_indexes[hand_key], dtype=np.intp)
        if qpos_indexes.shape != (16,):
            raise RuntimeError("Allegro qpos index mapping is not sixteen-dimensional")
        self._hand_qpos_indexes = qpos_indexes
        # Reset qpos is the only safe target before the first camera frame.
        self._held_hand_targets = np.clip(self.hand_qpos(), self.hand_lower_limits, self.hand_upper_limits)
        self._validate_hand_action_mapping()

    def _validate_hand_action_mapping(self) -> None:
        expected_low = np.asarray(self.hand.joint_targets_to_action(self.hand_lower_limits), dtype=np.float64)
        expected_high = np.asarray(self.hand.joint_targets_to_action(self.hand_upper_limits), dtype=np.float64)
        if expected_low.shape != (16,) or expected_high.shape != (16,):
            raise RuntimeError("Allegro target-to-action mapping has the wrong dimension")
        if (not np.all(np.isfinite(expected_low)) or not np.all(np.isfinite(expected_high))
                or np.any(expected_low < self._low[self.hand_slice] - 1e-8)
                or np.any(expected_high > self._high[self.hand_slice] + 1e-8)):
            raise RuntimeError("Allegro target-to-action mapping violates action_spec")
        round_trip = np.asarray(self.hand.action_to_joint_targets(expected_low), dtype=np.float64)
        if not np.allclose(round_trip, self.hand_lower_limits, atol=1e-8, rtol=0.0):
            raise RuntimeError("Allegro action mapping does not round-trip its lower limits")

    def hand_qpos(self) -> np.ndarray:
        """Return the current 16 physical joint positions in camera target order."""
        return np.asarray(self.env.sim.data.qpos[self._hand_qpos_indexes], dtype=np.float64).copy()

    @property
    def held_hand_targets(self) -> np.ndarray:
        """Return the last finite target retained across camera loss / staleness."""
        return self._held_hand_targets.copy()

    def hand_action_for_targets(self, joint_targets: Sequence[float] | None) -> np.ndarray:
        """Map valid absolute targets to the hand slice, otherwise hold safely.

        Holding means continuously commanding the most recently accepted joint
        target to the position actuators. It never falls back to an arbitrary
        normalized zero, which could move an asymmetric finger at startup or
        after camera loss.
        """
        if joint_targets is not None:
            values = np.asarray(joint_targets, dtype=np.float64)
            if values.shape == (16,) and np.all(np.isfinite(values)):
                self._held_hand_targets = np.clip(values, self.hand_lower_limits, self.hand_upper_limits)
        normalized = np.asarray(self.hand.joint_targets_to_action(self._held_hand_targets), dtype=np.float64)
        if normalized.shape != (16,) or not np.all(np.isfinite(normalized)):
            raise RuntimeError("Allegro target-to-action mapping returned an invalid action")
        return np.clip(normalized, self._low[self.hand_slice], self._high[self.hand_slice])

    def hold_action(self, base_action: Sequence[float] | None = None) -> np.ndarray:
        """A full action that leaves OSC neutral and holds every Allegro joint."""
        if base_action is None:
            action = self.neutral_action()
        else:
            action = np.asarray(base_action, dtype=np.float64).copy()
            if action.shape != (self.env.action_dim,) or not np.all(np.isfinite(action)):
                raise ValueError(f"base_action must be a finite vector of length {self.env.action_dim}")
        action[self.hand_slice] = self.hand_action_for_targets(None)
        return action

    def compose(
        self,
        physical_delta: Sequence[float],
        hand_targets: Sequence[float] | None = None,
        base_action: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return a valid 22-vector while updating only arm and hand slices."""
        action = self.hold_action(base_action)
        action[self.hand_slice] = self.hand_action_for_targets(hand_targets)
        return super().compose(physical_delta, action)

    def describe(self) -> dict[str, Any]:
        details = super().describe()
        details.update(
            hand_part=self.hand_name,
            hand_slice=[self.hand_slice.start, self.hand_slice.stop],
            hand_joint_names=list(self.hand_joint_names),
            hand_joint_limits={
                "lower": self.hand_lower_limits.tolist(),
                "upper": self.hand_upper_limits.tolist(),
            },
            hand_action_mapping="absolute radians -> normalized [-1, 1] -> MuJoCo position actuator ctrlrange",
            hand_tcp_site=self.hand.important_sites["grip_site"],
        )
        return details
