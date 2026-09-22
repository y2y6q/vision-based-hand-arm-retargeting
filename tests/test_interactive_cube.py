"""Regression tests for the Panda + Allegro interactive B-key cube helper."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.allegro_model import make_allegro_panda_env  # noqa: E402
from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.interactive_cube import InteractiveCubeScaler  # noqa: E402
from wrist_teleop.integrated import PandaAllegroActionComposer  # noqa: E402


class InteractiveCubeScalerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_allegro_panda_env(
            WristConfig(smoothing_alpha=None),
            has_renderer=False,
            has_offscreen_renderer=False,
            initialization_noise=None,
            seed=31,
        )
        self.addCleanup(self.env.close)
        self.env.reset()
        self.composer = PandaAllegroActionComposer(self.env)
        self.composer.initialize_goal()
        self.env.step(self.composer.compose(np.zeros(6), None))
        self.model = getattr(self.env.sim.model, "_model", self.env.sim.model)
        self.data = getattr(self.env.sim.data, "_data", self.env.sim.data)
        self.body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_main"))
        self.joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint0"))
        self.qpos_adr = int(self.model.jnt_qposadr[self.joint_id])
        self.qvel_adr = int(self.model.jnt_dofadr[self.joint_id])
        self.geom_ids = tuple(
            ident for ident in range(int(self.model.ngeom))
            if int(self.model.geom_bodyid[ident]) == self.body_id
        )

    def test_enlarge_updates_collision_visual_mass_and_inertia_without_resetting_scene(self):
        scaler = InteractiveCubeScaler(self.env, factor=1.5)
        self.assertTrue(scaler.available)
        original_sizes = np.asarray([self.model.geom_size[ident][:3] for ident in self.geom_ids], dtype=np.float64)
        original_mass = float(self.model.body_mass[self.body_id])
        original_inertia = self.model.body_inertia[self.body_id].copy()
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        time_before = float(self.data.time)

        record = scaler.enlarge()

        self.assertTrue(record["changed"])
        self.assertEqual(record["event"], "cube_size_changed")
        np.testing.assert_allclose(
            np.asarray([self.model.geom_size[ident][:3] for ident in self.geom_ids]),
            original_sizes * 1.5,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(self.model.body_mass[self.body_id], original_mass * 1.5 ** 3, atol=1.0e-12)
        np.testing.assert_allclose(self.model.body_inertia[self.body_id], original_inertia * 1.5 ** 5, atol=1.0e-12)
        # ``mj_setConst`` normally resets MjData. The helper restores the full
        # scene and changes only cube translation upward by the recorded amount.
        np.testing.assert_allclose(self.data.qpos[:self.qpos_adr], qpos_before[:self.qpos_adr], atol=1.0e-12)
        np.testing.assert_allclose(self.data.qpos[self.qpos_adr + 3:], qpos_before[self.qpos_adr + 3:], atol=1.0e-12)
        np.testing.assert_allclose(
            self.data.qpos[self.qpos_adr:self.qpos_adr + 2], qpos_before[self.qpos_adr:self.qpos_adr + 2], atol=1.0e-12
        )
        self.assertGreater(self.data.qpos[self.qpos_adr + 2], qpos_before[self.qpos_adr + 2])
        np.testing.assert_allclose(self.data.qvel, qvel_before, atol=0.0)
        self.assertEqual(float(self.data.time), time_before)
        self.assertTrue(record["velocity_preserved"])
        self.assertEqual(record["physics_refresh"], "mj_setConst+mj_forward")

        for _ in range(5):
            self.env.step(self.composer.compose(np.zeros(6), None))
        self.assertTrue(np.all(np.isfinite(self.data.qpos)))
        self.assertTrue(np.all(np.isfinite(self.data.qvel)))

    def test_repeat_is_idempotent_and_restore_default_precedes_r_reset(self):
        scaler = InteractiveCubeScaler(self.env, factor=1.5)
        original_sizes = np.asarray([self.model.geom_size[ident][:3] for ident in self.geom_ids], dtype=np.float64)
        scaler.enlarge()
        self.assertFalse(scaler.enlarge()["changed"])
        self.assertEqual(scaler.enlarge()["reason"], "already_enlarged")

        restored = scaler.restore_default()
        self.assertTrue(restored["changed"])
        np.testing.assert_allclose(
            np.asarray([self.model.geom_size[ident][:3] for ident in self.geom_ids]), original_sizes, atol=1.0e-12
        )
        self.env.reset()
        reset_scaler = InteractiveCubeScaler(self.env, factor=1.5)
        self.assertFalse(reset_scaler.enlarged)
        self.assertEqual(reset_scaler.status()["current_factor"], 1.0)


if __name__ == "__main__":
    unittest.main()
