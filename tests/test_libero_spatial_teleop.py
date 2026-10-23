"""LIBERO Spatial adapter regressions. Run with .venv-libero (robosuite 1.4):

    .\\.venv-libero\\Scripts\\python.exe -m pytest -q .\\tests\\test_libero_spatial_teleop.py

Skipped automatically in the robosuite 1.5 mainline interpreter.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import robosuite

    LEGACY_ROBOSUITE = robosuite.__version__.startswith("1.4")
except ImportError:  # pragma: no cover
    LEGACY_ROBOSUITE = False


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO adapter tests require .venv-libero (robosuite 1.4)")
class LiberoShadowTeleopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from wrist_teleop.config import WristConfig
        from wrist_teleop import libero_spatial_teleop as L

        cls.L = L
        cls.config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
        cls.session = L.OfficialLiberoSpatialSession(
            L.PANDA_SHADOW_SPEC, cls.config, task_id=0, task_order=0, init_state_id=0, seed=7,
            has_offscreen_renderer=True,
        )
        cls.session.open()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.close()

    def _composer(self):
        self.session.reset()
        composer = self.L.LiberoActionComposer(self.session.env, self.L.PANDA_SHADOW_SPEC)
        composer.initialize_goal()
        return composer

    def test_shadow_uses_flange_rigid_tcp(self):
        composer = self._composer()
        self.assertEqual(composer.gripper.describe()["gripper_type"], "ShadowHandRightRigidTcp")
        sim = self.session.env.sim
        body = sim.model.site_bodyid[sim.model.site_name2id(composer.controller.eef_name)]
        self.assertEqual(sim.model.body_id2name(int(sim.model.body_parentid[body])), "gripper0_rh_mount")

    def test_zero_input_holds_pose_while_fingers_open_and_close(self):
        composer = self._composer()
        lower, upper = composer.gripper.lower_limits, composer.gripper.upper_limits
        start, start_ori = composer.ee_pose()
        for step in range(80):
            hand = lower + 0.9 * (upper - lower) if (step // 10) % 2 else np.clip(np.zeros(20), lower, upper)
            hand = hand.copy()
            hand[:2] = 0.0
            action = composer.compose(np.zeros(6), hand)
            self.session.env.step(action)
        end, end_ori = composer.ee_pose()
        self.assertLess(np.linalg.norm(end - start), 0.005)  # was ~110 mm before the fix
        angle = np.degrees(np.arccos(np.clip((np.trace(start_ori.T @ end_ori) - 1.0) / 2.0, -1.0, 1.0)))
        self.assertLess(angle, 1.0)

    def test_full_deflection_speed_and_release_stop(self):
        composer = self._composer()
        step = self.config.max_translation_delta_m * self.L.LIBERO_ARM_TRANSLATION_SPEED_SCALE
        start = composer.ee_pose()[0]
        for _ in range(20):
            self.session.env.step(composer.compose(np.array([step, 0.0, 0.0, 0.0, 0.0, 0.0])))
        moved = composer.ee_pose()[0]
        self.assertGreater(np.linalg.norm(moved - start), 0.15)  # >= 0.15 m in 1 s
        for _ in range(20):
            action = composer.compose(np.zeros(6))
            self.assertTrue(np.all(np.abs(action[:6]) <= composer.high[:6] + 1e-12))
            self.session.env.step(action)
        self.assertLess(np.linalg.norm(composer.ee_pose()[0] - moved), 0.01)

    def test_reset_rebinds_laser_to_new_sim(self):
        from wrist_teleop.visualization import LaserPointer

        laser_config = self.L._shadow_laser_config(self.config)
        stale = LaserPointer(self.session.env, laser_config)
        self.session.reset()  # robosuite 1.4 hard reset frees the old MjSim
        with self.assertRaises(AttributeError):
            stale.update()
        fresh = LaserPointer(self.session.env, laser_config)
        composer = self.L.LiberoActionComposer(self.session.env, self.L.PANDA_SHADOW_SPEC)
        composer.initialize_goal()
        self.session.env.step(composer.compose(np.zeros(6)))
        self.assertTrue(fresh.update()["active"])


if __name__ == "__main__":
    unittest.main()
