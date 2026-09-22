"""Focused headless regression coverage for the custom robosuite Allegro hand."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.allegro_model import (  # noqa: E402
    ALLEGRO_JOINT_NAMES,
    AllegroRightHand,
    GRIPPER_TYPE,
    make_allegro_panda_env,
)
from wrist_teleop.config import WristConfig  # noqa: E402


class AllegroModelTests(unittest.TestCase):
    def test_source_model_has_exact_16_position_joints_and_collision_geometry(self):
        hand = AllegroRightHand(idn="unit")
        self.assertEqual(hand.dof, 16)
        self.assertEqual(tuple(hand._joints), ALLEGRO_JOINT_NAMES)
        self.assertEqual(len(hand.actuators), 16)
        self.assertEqual(len(hand.contact_geoms), 23)
        self.assertGreater(len(hand.visual_geoms), 0)
        self.assertEqual(hand.init_qpos[12], hand.lower_limits[12])
        np.testing.assert_allclose(
            hand.action_to_joint_targets(hand.initial_action), hand.init_qpos, atol=1e-12
        )

        model = hand.get_model()
        self.assertEqual(model.njnt, 16)
        self.assertEqual(model.nu, 16)
        self.assertTrue(np.all(model.geom_contype[np.asarray([model.geom(name).id for name in hand.contact_geoms])] == 1))
        np.testing.assert_allclose(model.actuator_ctrlrange[:, 0], hand.lower_limits)
        np.testing.assert_allclose(model.actuator_ctrlrange[:, 1], hand.upper_limits)


class AllegroPandaIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_allegro_panda_env(WristConfig(smoothing_alpha=None), has_renderer=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()
        self.robot = self.env.robots[0]
        self.hand = self.robot.gripper["right"]
        self.hand_slice = slice(*self.robot.composite_controller._action_split_indexes["right_gripper"])

    def _action_for_targets(self, targets: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.action_dim, dtype=np.float64)
        action[self.hand_slice] = self.hand.joint_targets_to_action(targets)
        return action

    def test_registered_hand_replaces_panda_gripper_and_exposes_22d_layout(self):
        self.assertEqual(GRIPPER_TYPE, type(self.hand).__name__)
        self.assertEqual(self.env.action_dim, 22)
        self.assertEqual(dict(self.robot.composite_controller._action_split_indexes)["right"], (0, 6))
        self.assertEqual((self.hand_slice.start, self.hand_slice.stop), (6, 22))
        self.assertEqual(self.hand.dof, 16)
        self.assertNotIn("PandaGripper", type(self.hand).__name__)
        self.assertGreaterEqual(self.env.sim.model.site_name2id(self.hand.important_sites["grip_site"]), 0)

    def test_transparent_palm_laser_site_marks_the_source_palm_frame(self):
        site_id = self.env.sim.model.site_name2id("gripper0_right_palm_laser_site")
        palm_body_id = self.env.sim.model.body_name2id("gripper0_right_palm")
        self.assertGreaterEqual(site_id, 0)
        self.assertEqual(int(self.env.sim.model.site_bodyid[site_id]), palm_body_id)
        np.testing.assert_allclose(self.env.sim.model.site_rgba[site_id], [1.0, 0.0, 0.0, 0.0])

    def test_normalized_hand_slice_drives_exact_position_actuator_targets(self):
        targets = self.hand.init_qpos.copy()
        targets[0] += 0.2
        targets[6] += 0.4
        action = self._action_for_targets(targets)
        self.env.step(action)
        actuator_ids = self.robot._ref_joint_gripper_actuator_indexes["right"]
        np.testing.assert_allclose(self.env.sim.data.ctrl[actuator_ids], targets, atol=1e-12)
        # The arm remains the all-zero OSC command while the hand moves.
        np.testing.assert_array_equal(action[:6], np.zeros(6))

    def test_single_joint_sweep_moves_the_selected_finger_without_instability(self):
        qpos_ids = self.robot._ref_gripper_joint_pos_indexes["right"]
        start = self.env.sim.data.qpos[qpos_ids].copy()
        targets = self.hand.init_qpos.copy()
        targets[0] += 0.2
        action = self._action_for_targets(targets)
        for _ in range(100):
            self.env.step(action)
        final = self.env.sim.data.qpos[qpos_ids]
        self.assertGreater(final[0] - start[0], 0.15)
        self.assertLess(abs(final[0] - targets[0]), 0.01)
        self.assertTrue(np.all(np.isfinite(self.env.sim.data.qpos)))
        self.assertTrue(np.all(np.isfinite(self.env.sim.data.qvel)))

    def test_initial_hold_is_finite_and_has_no_hand_self_contact(self):
        action = self._action_for_targets(self.hand.init_qpos)
        for _ in range(20):
            self.env.step(action)
        self.assertTrue(np.all(np.isfinite(self.env.sim.data.qpos)))
        # Connected link pairs are excluded to prevent reset jumps. This does
        # not suppress contact with external objects / table geometry.
        for index in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[index]
            first = self.env.sim.model.geom_id2name(int(contact.geom1)) or ""
            second = self.env.sim.model.geom_id2name(int(contact.geom2)) or ""
            self.assertFalse(first.startswith("gripper0_right_") and second.startswith("gripper0_right_"))

    def test_hand_collision_geometry_reports_contact_with_lift_cube(self):
        # Place the scene's existing free cube at a fingertip collision geom,
        # then ask MuJoCo to recompute contacts. This checks actual external
        # collision participation without claiming that a grasp was achieved.
        finger_geom = next(name for name in self.hand.contact_geoms if "link_3.0_tip" in name)
        finger_position = self.env.sim.data.geom_xpos[self.env.sim.model.geom_name2id(finger_geom)].copy()
        cube_start, cube_stop = self.env.sim.model.get_joint_qpos_addr("cube_joint0")
        self.env.sim.data.qpos[cube_start:cube_start + 3] = finger_position
        self.env.sim.data.qpos[cube_start + 3:cube_stop] = [1.0, 0.0, 0.0, 0.0]
        self.env.sim.forward()
        pairs = []
        for index in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[index]
            first = self.env.sim.model.geom_id2name(int(contact.geom1)) or ""
            second = self.env.sim.model.geom_id2name(int(contact.geom2)) or ""
            pairs.append((first, second))
        self.assertTrue(any(
            (first.startswith("gripper0_right_") and second.startswith("cube_"))
            or (second.startswith("gripper0_right_") and first.startswith("cube_"))
            for first, second in pairs
        ))


if __name__ == "__main__":
    unittest.main()
