"""Actual robosuite 1.5.2 OSC_POSE composition checks."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_teleop import OscPoseComposer, SixDofMapper, WristConfig, create_panda_osc_pose_env


class OscPoseComposerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = WristConfig(smoothing_alpha=None)
        cls.env = create_panda_osc_pose_env(cls.config, has_renderer=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()
        self.composer = OscPoseComposer(self.env)
        self.composer.initialize_goal()

    def test_installed_action_layout_and_world_delta_mode(self):
        detail = self.composer.describe()
        self.assertEqual(detail["action_dim"], 7)
        self.assertEqual(detail["arm_slice"], [0, 6])
        self.assertEqual(detail["action_slices"]["right_gripper"], [6, 7])
        self.assertEqual(detail["input_type"], "delta")
        self.assertEqual(detail["input_ref_frame"], "world")

    def test_composer_changes_only_six_arm_entries(self):
        base = np.array([0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77])
        physical = np.array([0.002, -0.004, 0.001, 0.02, -0.04, 0.01])
        action = self.composer.compose(physical, base)
        np.testing.assert_allclose(action[:6], [0.5, -1.0, 0.25, 0.5, -1.0, 0.25])
        self.assertEqual(action[6], base[6])

    def test_zero_action_holds_desired_goal_and_gripper_state(self):
        controller = self.composer.controller
        pose = self.composer.ee_pose()
        # The first format_action expands Panda's one policy DoF to its two
        # physical fingers. Compare subsequent zero / hold actions.
        self.env.step(self.composer.neutral_action())
        gripper_action = self.env.robots[0].gripper["right"].current_action.copy()
        for _ in range(10):
            self.env.step(self.composer.neutral_action())
        np.testing.assert_allclose(controller.goal_pos, pose["position_m"], atol=1e-10)
        np.testing.assert_allclose(self.env.robots[0].gripper["right"].current_action, gripper_action)
        np.testing.assert_allclose(self.composer.ee_pose()["position_m"], pose["position_m"], atol=1e-4)

    def test_translation_and_rotation_commands_are_separate(self):
        controller = self.composer.controller
        goal_pos = controller.goal_pos.copy()
        goal_ori = controller.goal_ori.copy()
        self.env.step(self.composer.compose([0.002, 0, 0, 0, 0, 0]))
        np.testing.assert_allclose(controller.goal_pos - goal_pos, [0.002, 0, 0], atol=1e-12)
        np.testing.assert_allclose(controller.goal_ori, goal_ori, atol=1e-12)
        goal_pos, goal_ori = controller.goal_pos.copy(), controller.goal_ori.copy()
        self.env.step(self.composer.compose([0, 0, 0, 0, 0.02, 0]))
        np.testing.assert_allclose(controller.goal_pos, goal_pos, atol=1e-12)
        self.assertFalse(np.allclose(controller.goal_ori, goal_ori))

    def test_simulated_spacemouse_motion_reaches_only_osc_arm_slice(self):
        mapper = SixDofMapper(WristConfig(smoothing_alpha=None))
        physical = mapper.map([350, 0, 0, 0, 0, 0])
        action = self.composer.compose(physical)
        self.assertGreater(np.linalg.norm(action[self.composer.arm_slice]), 0.0)
        # The mapper has no button input and the composer leaves this Panda
        # gripper slice at zero / hold for every simulated SpaceMouse motion.
        np.testing.assert_array_equal(action[6:], np.zeros(1))

    def test_button_configuration_never_changes_the_osc_arm_action(self):
        # The integrated hand gesture controller consumes buttons separately.
        # This wrist-only composer accepts no button state and therefore cannot
        # turn a button press into an arm action or a process exit binding.
        self.assertEqual(WristConfig().hand_gestures.left_button_index, 0)
        self.assertEqual(WristConfig().hand_gestures.right_button_index, 1)
        mapper = SixDofMapper(WristConfig(smoothing_alpha=None))
        command = mapper.map([0, 0, 0, 0, 0, 0])
        action = self.composer.compose(command)
        np.testing.assert_array_equal(action, np.zeros(self.env.action_dim))
