"""Regression checks for the shared read-only interactive grasp telemetry."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.allegro_model import make_allegro_panda_env  # noqa: E402
from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.grasp_telemetry import GraspTelemetry, GraspTelemetryConfig  # noqa: E402
from wrist_teleop.integrated import PandaAllegroActionComposer  # noqa: E402


class GraspTelemetryTests(unittest.TestCase):
    def test_real_lift_cube_readout_does_not_change_physics_and_reports_waiting_failure(self):
        config = WristConfig(smoothing_alpha=None)
        env = make_allegro_panda_env(
            config,
            has_renderer=False,
            has_offscreen_renderer=False,
            initialization_noise=None,
            seed=23,
        )
        self.addCleanup(env.close)
        env.reset()
        composer = PandaAllegroActionComposer(env)
        composer.initialize_goal()
        telemetry = GraspTelemetry(env, config.grasp_telemetry)
        before_qpos, before_qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
        initial = telemetry.update()
        np.testing.assert_array_equal(env.sim.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.sim.data.qvel, before_qvel)
        self.assertTrue(initial["available"])
        self.assertFalse(initial["passed"])
        env.step(composer.compose(np.zeros(6), None))
        step = telemetry.update()
        self.assertIn("table_contact", step)
        self.assertIn("hand_contact_geoms", step)
        summary = telemetry.summary()
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["failure_reason"], "cube_not_lifted")

    def test_config_requires_positive_common_acceptance_thresholds(self):
        with self.assertRaises(ValueError):
            GraspTelemetryConfig(lift_height_m=0.0)
        config = WristConfig.from_dict(
            {"grasp_telemetry": {"lift_height_m": 0.04, "hold_duration_s": 3.2, "max_penetration_m": 0.003}}
        )
        self.assertEqual(config.grasp_telemetry.lift_height_m, 0.04)
        self.assertEqual(config.grasp_telemetry.hold_duration_s, 3.2)


if __name__ == "__main__":
    unittest.main()
