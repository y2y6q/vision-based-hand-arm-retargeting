"""Camera-only LIBERO arm control in the real scene. Run with .venv-libero:

    .\\.venv-libero\\Scripts\\python.exe -m pytest -q .\\tests\\test_libero_camera_teleop.py
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


def _lead(composer) -> np.ndarray:
    from wrist_teleop.camera_arm_control import matrix_to_rotvec

    pos, ori = composer.ee_pose()
    return np.concatenate((composer.goal_pos - pos, matrix_to_rotvec(composer.goal_ori @ ori.T)))


@unittest.skipUnless(LEGACY_ROBOSUITE, "requires .venv-libero (robosuite 1.4)")
class LiberoCameraArmTests(unittest.TestCase):
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

    def _fresh(self):
        from wrist_teleop.camera_arm_control import ArmSafetyGuard
        from wrist_teleop.libero_camera_teleop import build_workspace

        self.session.reset()
        env = self.session.env
        composer = self.L.LiberoActionComposer(env, self.L.PANDA_SHADOW_SPEC)
        composer.initialize_goal()
        guard = ArmSafetyGuard(env.sim, env.robots[0], composer.controller.eef_name)
        return env, composer, guard, build_workspace(env, composer)

    def test_synthetic_hand_moves_robot_in_operator_directions(self):
        from wrist_teleop.camera_arm_control import (CameraArmMapper, PalmShapeTemplate, TargetFollower,
                                                     estimate_hand_pose)
        from wrist_teleop.libero_camera_teleop import MAX_ROT_SPEED_RAD_S, MAX_SPEED_M_S, arm_step
        from wrist_teleop.mapper import SixDofMapper
        from wrist_teleop.synthetic_hand import control_offset_to_camera, local_rotation_to_camera, synthesize

        env, composer, guard, limits = self._fresh()
        basis = SixDofMapper(self.config).control_rotation
        mapper, template = CameraArmMapper(), PalmShapeTemplate()
        follower = TargetFollower(MAX_SPEED_M_S, MAX_ROT_SPEED_RAD_S)
        rng = np.random.default_rng(0)
        start, t = np.array((0.0, 0.03, 0.5)), 0.0
        hand = {"offset": np.zeros(3), "knob": 0.0}

        def hold(offset, knob_deg, seconds, ramp_s=0.8):
            nonlocal t
            begin_offset, begin_knob = hand["offset"], hand["knob"]
            offset = np.asarray(offset, dtype=float)
            ticks = int(seconds * 20)
            for tick in range(ticks):
                frac = min(1.0, (tick + 1) / max(ramp_s * 20, 1.0))
                for _sub in range(2):
                    d = synthesize(start + control_offset_to_camera(begin_offset + frac * (offset - begin_offset)),
                                   local_rotation_to_camera(np.radians((0.0, 0.0, begin_knob + frac * (knob_deg - begin_knob)))),
                                   image_noise=1 / 640, world_noise_m=0.002, rng=rng)
                    mapper.observe(estimate_hand_pose(d.image_landmarks, d.world_landmarks,
                                                      frame_shape=d.frame_shape, mirrored=True,
                                                      focal_width_normalized=0.78, timestamp_s=t,
                                                      palm_template=template), t)
                    t += 0.025
                measured = composer.ee_pose()
                goal = (composer.goal_pos, composer.goal_ori)
                status = mapper.target(*goal, basis, t)
                delta = np.zeros(6)
                if status.state == "tracking":
                    delta, _, _, _ = arm_step(mapper, status.target_pos, status.target_ori, limits=limits,
                                              follower=follower, guard=guard, goal=goal, measured=measured, dt=0.05)
                env.step(composer.compose(delta, None))
            hand["offset"], hand["knob"] = offset, float(knob_deg)

        hold((0, 0, 0), 0.0, 2.0)
        reference = composer.ee_pose()

        def moved():
            return (composer.ee_pose()[0] - reference[0]) @ basis  # [right, away, up]

        hold((0.08, 0, 0), 0.0, 2.0)  # your right -> right
        right = moved()
        self.assertGreater(right[0], 0.04)
        np.testing.assert_allclose(right[1:], 0.0, atol=0.012)
        hold((0.08, 0.07, 0), 0.0, 2.0, ramp_s=0.3)  # held nearer the camera -> away
        away = moved()
        self.assertGreater(away[1], right[1] + 0.03)
        hold((0.08, 0.0, 0), 0.0, 1.0, ramp_s=0.3)  # neutral again: stops
        stopped = moved()
        hold((0.08, 0.0, 0), 0.0, 1.0)
        self.assertLess(float(np.linalg.norm(moved() - stopped)), 0.006)
        hold((0.08, 0.0, 0.06), 0.0, 2.0)  # up -> up
        self.assertGreater(moved()[2], stopped[2] + 0.03)
        before = composer.ee_pose()[1]
        hold((0.08, 0.0, 0.06), 30.0, 2.5)  # knob turn -> yaw
        relative = composer.ee_pose()[1] @ before.T
        self.assertAlmostEqual(np.degrees(np.arctan2(relative[1, 0], relative[0, 0])), 30.0, delta=4.0)
        still = []
        for _ in range(3):
            hold((0.08, 0.0, 0.06), 30.0, 0.5)
            still.append(composer.ee_pose()[0])
        self.assertLess(float(np.ptp(np.array(still), axis=0).max()), 0.002)

    def test_virtual_wall_lifts_when_backing_off_or_when_safe_again(self):
        env, composer, _, _ = self._fresh()
        from wrist_teleop.camera_arm_control import ArmSafetyGuard

        now = {"t": 0.0}
        guard = ArmSafetyGuard(env.sim, env.robots[0], composer.controller.eef_name, clock=lambda: now["t"])
        position = np.array(env.sim.data.site_xpos[guard.site])
        push = np.array((0.0, 0.0, -0.01, 0, 0, 0))
        guard._wall, guard._wall_checked_at = (position, np.array((0.0, 0.0, -1.0))), 0.0
        held, reasons, info = guard.check(push)
        self.assertIn("virtual_wall", reasons)  # within the re-check period: still walled
        self.assertAlmostEqual(float(np.linalg.norm(held[:3])), 0.0)
        self.assertIn("wall_direction", info)
        now["t"] = 1.1  # re-check: this pose is far from any limit, so the wall lifts
        free, reasons, _ = guard.check(push)
        self.assertEqual(reasons, [])
        np.testing.assert_allclose(free, push)
        guard._wall = (position, np.array((0.0, 0.0, -1.0)))
        back, reasons, _ = guard.check(-push)  # commanding away lifts it at once
        self.assertEqual(reasons, [])
        self.assertIsNone(guard.wall_direction)

    def test_guard_stops_at_joint_limit_and_lets_operator_back_out(self):
        env, composer, guard, _ = self._fresh()
        blocked = False
        for _ in range(200):  # keep twisting the wrist: ~0.06 rad per step
            delta, reasons, info = guard.check(np.array((0, 0, 0, 0, 0, 0.06)), _lead(composer))
            blocked = blocked or any(r.startswith("joint_limit") for r in reasons)
            env.step(composer.compose(delta, None))
        self.assertTrue(blocked)
        _, _, info = guard.check(np.zeros(6))
        self.assertGreater(info["min_joint_margin_rad"], 0.5 * guard.joint_margin)
        back, reasons, _ = guard.check(np.array((0, 0, 0, 0, 0, -0.06)), _lead(composer))
        self.assertGreater(float(np.linalg.norm(back)), 0.03)  # reversing is allowed

    def test_guard_keeps_elbow_bent_and_lets_operator_back_out(self):
        env, composer, guard, _ = self._fresh()
        base = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")])
        outward = composer.ee_pose()[0] - base
        outward[2] = 0.0
        outward /= np.linalg.norm(outward)
        ratios = []
        for _ in range(250):  # keep pushing straight out at 0.2 m/s, no workspace limits
            delta, reasons, info = guard.check(np.concatenate((0.01 * outward, np.zeros(3))), _lead(composer))
            ratios.append(info["manipulability_ratio"])
            env.step(composer.compose(delta, None))
        elbow = env.sim.data.qpos[guard.qpos_index][3]
        self.assertLess(elbow, guard.upper[3] - 0.5 * guard.upper_margin[3])  # elbow kept bent
        self.assertGreater(min(ratios), 0.5 * guard.min_manipulability / guard.reference_manipulability)
        back, _, _ = guard.check(np.concatenate((-0.01 * outward, np.zeros(3))), _lead(composer))
        self.assertGreater(float(np.linalg.norm(back)), 0.005)

    def test_workspace_limits_keep_far_reach_well_conditioned(self):
        env, composer, guard, limits = self._fresh()
        from wrist_teleop.camera_arm_control import TargetFollower

        follower = TargetFollower()
        far = composer.ee_pose()[0] + np.array((0.6, 0.0, 0.0))
        lowest = 10.0
        for _ in range(150):
            pos, ori = composer.ee_pose()
            target, target_ori, _ = limits.clamp(far, ori)
            delta = follower.delta(target, target_ori, composer.goal_pos, composer.goal_ori, 0.05)
            delta, _, info = guard.check(delta, _lead(composer))
            lowest = min(lowest, info["manipulability_ratio"])
            env.step(composer.compose(delta, None))
        self.assertGreater(lowest, 0.5)


if __name__ == "__main__":
    unittest.main()
