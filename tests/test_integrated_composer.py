"""Real robosuite regression checks for independent wrist / Allegro actions."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop import PandaAllegroActionComposer, SixDofMapper, WristConfig  # noqa: E402
from wrist_teleop.allegro_model import make_allegro_panda_env  # noqa: E402
from wrist_teleop.hand_input import (  # noqa: E402
    DEFAULT_ALLEGRO_JOINT_NAMES,
    DEFAULT_ALLEGRO_LOWER_LIMITS,
    DEFAULT_ALLEGRO_UPPER_LIMITS,
    curls_to_allegro_targets,
)
from wrist_teleop.hand_gestures import AllegroHandGestureController  # noqa: E402


class IntegratedComposerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = WristConfig(smoothing_alpha=None)
        cls.env = make_allegro_panda_env(cls.config, has_renderer=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()
        self.composer = PandaAllegroActionComposer(self.env)
        self.composer.initialize_goal()

    def test_live_action_layout_and_camera_joint_contract(self):
        detail = self.composer.describe()
        self.assertEqual(detail["action_dim"], 22)
        self.assertEqual(detail["arm_slice"], [0, 6])
        self.assertEqual(detail["hand_slice"], [6, 22])
        self.assertEqual(tuple(detail["hand_joint_names"]), DEFAULT_ALLEGRO_JOINT_NAMES)
        np.testing.assert_allclose(self.composer.hand_lower_limits, DEFAULT_ALLEGRO_LOWER_LIMITS)
        np.testing.assert_allclose(self.composer.hand_upper_limits, DEFAULT_ALLEGRO_UPPER_LIMITS)

    def test_arm_and_hand_commands_are_strictly_slice_isolated(self):
        target = curls_to_allegro_targets(
            [0.7, 0.5, 0.3, 0.8], self.composer.hand_lower_limits, self.composer.hand_upper_limits
        )
        arm_delta = np.array([0.002, -0.001, 0.0, 0.0, 0.02, 0.0])
        action = self.composer.compose(arm_delta, target)
        expected_arm = np.array([0.5, -0.25, 0.0, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(action[self.composer.arm_slice], expected_arm)
        np.testing.assert_allclose(action[self.composer.hand_slice], self.composer.hand.joint_targets_to_action(target))

        # A hand-only update leaves every wrist entry at its neutral zero.
        hand_only = self.composer.compose(np.zeros(6), target)
        np.testing.assert_array_equal(hand_only[self.composer.arm_slice], np.zeros(6))
        # An arm-only update reuses, but cannot alter, the last camera target.
        arm_only = self.composer.compose(arm_delta, None)
        np.testing.assert_allclose(arm_only[self.composer.hand_slice], hand_only[self.composer.hand_slice])

    def test_invalid_camera_target_holds_last_safe_position_command(self):
        target = curls_to_allegro_targets(
            [1.0, 0.4, 0.0, 0.6], self.composer.hand_lower_limits, self.composer.hand_upper_limits
        )
        accepted = self.composer.compose(np.zeros(6), target)
        retained = self.composer.compose(np.zeros(6), np.full(16, np.nan))
        np.testing.assert_allclose(retained[self.composer.hand_slice], accepted[self.composer.hand_slice])
        np.testing.assert_allclose(self.composer.held_hand_targets, target)

    def test_simulated_spacemouse_and_camera_can_share_one_action(self):
        mapper = SixDofMapper(self.config)
        wrist = mapper.map([350, 350, 350, 350, 350, 350])
        target = curls_to_allegro_targets(
            [0.2, 0.4, 0.6, 0.8], self.composer.hand_lower_limits, self.composer.hand_upper_limits
        )
        action = self.composer.compose(wrist, target)
        self.assertGreater(np.linalg.norm(action[self.composer.arm_slice]), 0.0)
        self.assertGreater(np.linalg.norm(action[3:6]), 0.0)
        self.assertEqual(action.shape, (22,))
        self.env.step(action)
        self.assertTrue(np.all(np.isfinite(self.composer.hand_qpos())))

    def test_button_gesture_changes_only_hand_slice_and_axes_only_arm_slice(self):
        gestures = AllegroHandGestureController(
            joint_names=self.composer.hand_joint_names,
            lower_limits=self.composer.hand_lower_limits,
            upper_limits=self.composer.hand_upper_limits,
        )
        camera_target = (self.composer.hand_lower_limits + self.composer.hand_upper_limits) * 0.5
        gestures.update((False, False), camera_target, camera_valid=True, timestamp_s=1.0)
        button_hand = gestures.update((True, False), camera_target, camera_valid=True, timestamp_s=1.3)
        mapper = SixDofMapper(WristConfig(smoothing_alpha=None))
        wrist = mapper.map([350, 0, 0, 0, 0, 0])
        with_button = self.composer.compose(wrist, button_hand.joint_targets)
        hand_only = self.composer.compose(np.zeros(6), button_hand.joint_targets)
        np.testing.assert_allclose(with_button[self.composer.arm_slice],
                                   self.composer.compose(wrist, None)[self.composer.arm_slice])
        np.testing.assert_allclose(with_button[self.composer.hand_slice], hand_only[self.composer.hand_slice])


if __name__ == "__main__":
    unittest.main()
