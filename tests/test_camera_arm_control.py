"""Camera-only arm control: wrist pose estimate, mapping, filtering, limits (no robot)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.camera_arm_control import (  # noqa: E402
    CameraArmHandInput,
    CameraArmMapper,
    CameraArmSettings,
    PalmShapeTemplate,
    TargetFollower,
    WorkspaceLimits,
    estimate_hand_pose,
    matrix_to_rotvec,
    rotvec_to_matrix,
)
from wrist_teleop.synthetic_hand import (  # noqa: E402
    control_offset_to_camera,
    control_rotation_to_camera,
    local_rotation_to_camera,
    synthesize,
)

START = np.array((0.0, 0.03, 0.50))


def _pose(detection, template, t=0.0):
    return estimate_hand_pose(detection.image_landmarks, detection.world_landmarks, frame_shape=detection.frame_shape,
                              mirrored=True, focal_width_normalized=0.78, timestamp_s=t, palm_template=template)


class _Runner:
    """Feed a synthetic hand at 30 Hz into a mapper (identity world basis)."""

    def __init__(self, settings=None, noise_px=1.0, world_noise_m=0.002, seed=0):
        self.mapper = CameraArmMapper(settings or CameraArmSettings())
        self.template = PalmShapeTemplate()
        self.rng = np.random.default_rng(seed)
        self.noise, self.world_noise = noise_px / 640.0, world_noise_m
        self.t = 0.0
        self.offset, self.rot = np.zeros(3), np.zeros(3)
        self.robot_pos, self.robot_ori = np.zeros(3), np.eye(3)

    def run(self, offset=None, rotdeg=None, seconds=1.0, ramp_s=0.8, frame=None):
        begin_o, begin_r = self.offset.copy(), self.rot.copy()
        end_o = begin_o if offset is None else np.asarray(offset, float)
        end_r = begin_r if rotdeg is None else np.radians(np.asarray(rotdeg, float))
        status = None
        frames = int(seconds * 30)
        for k in range(frames):
            frac = min(1.0, (k + 1) / max(ramp_s * 30, 1))
            self.offset, self.rot = begin_o + frac * (end_o - begin_o), begin_r + frac * (end_r - begin_r)
            detection = synthesize(START + control_offset_to_camera(self.offset), local_rotation_to_camera(self.rot),
                                   image_noise=self.noise, world_noise_m=self.world_noise, rng=self.rng)
            if frame is not None:
                detection = frame(detection, k) or detection
            try:
                self.mapper.observe(_pose(detection, self.template, self.t), self.t)
            except ValueError:
                self.mapper.observe(None, self.t)
            status = self.mapper.target(self.robot_pos, self.robot_ori, np.eye(3), self.t)
            self.t += 1.0 / 30.0
        return status


class EstimatorTests(unittest.TestCase):
    def test_perspective_wrist_estimate_is_exact_off_centre_and_rotated(self):
        for offset in ((0.0, 0.0, 0.0), (0.12, 0.05, 0.06), (-0.15, -0.08, -0.05)):
            for rotation in ((0, 0, 0), (0, 0, 35), (30, 0, 0), (0, 25, 0), (20, -20, 40)):
                position = START + control_offset_to_camera(np.array(offset))
                rot = control_rotation_to_camera(np.radians(rotation))
                template = PalmShapeTemplate()
                for _ in range(31):
                    pose = _pose(synthesize(position, rot), template)
                np.testing.assert_allclose(pose.position_m, position, atol=1e-3)
                self.assertTrue(pose.ready)

    def test_palm_cut_by_border_is_rejected(self):
        with self.assertRaises(ValueError):
            _pose(synthesize(START + control_offset_to_camera(np.array((-0.32, 0.0, 0.0))), np.eye(3)), None)

    def test_hand_near_the_border_still_tracks_and_reports_it(self):
        centre = _pose(synthesize(START, np.eye(3)), None)
        near = _pose(synthesize(START + control_offset_to_camera(np.array((0.24, 0.0, 0.0))), np.eye(3)), None)
        self.assertGreater(centre.edge_distance, 0.2)
        self.assertLess(near.edge_distance, CameraArmSettings().edge_warning)


class MappingTests(unittest.TestCase):
    def test_directions_and_pointer_acceleration(self):
        runner = _Runner()
        self.assertEqual(runner.run(seconds=2.0).state, "tracking")
        slow = runner.run(offset=(0.08, 0.0, 0.0), seconds=3.0, ramp_s=2.5).target_pos  # your right -> right
        self.assertGreater(slow[0], 0.03)
        np.testing.assert_allclose(slow[1:], 0.0, atol=0.003)
        runner = _Runner()
        runner.run(seconds=2.0)
        fast = runner.run(offset=(0.08, 0.0, 0.0), seconds=1.5, ramp_s=0.4).target_pos
        self.assertGreater(fast[0], 1.6 * slow[0])  # same hand travel, faster -> farther
        up = runner.run(offset=(0.08, 0.0, 0.06), seconds=1.5, ramp_s=0.4).target_pos  # up -> up
        self.assertGreater(up[2], 0.04)
        self.assertAlmostEqual(up[0], fast[0], delta=0.003)

    def test_depth_is_rate_controlled_with_a_deadzone(self):
        runner = _Runner()
        runner.run(seconds=2.0)
        inside = runner.run(offset=(0.0, 0.02, 0.0), seconds=1.5, ramp_s=0.3).target_pos
        self.assertAlmostEqual(inside[1], 0.0, delta=0.002)  # inside the deadzone: no motion
        half = runner.run(offset=(0.0, 0.07, 0.0), seconds=1.0, ramp_s=0.3).target_pos  # nearer -> away
        held = runner.run(seconds=1.0).target_pos
        self.assertGreater(half[1], 0.01)
        self.assertGreater(held[1], half[1] + 0.02)  # keeps moving while held
        back = runner.run(offset=(0.0, 0.0, 0.0), seconds=1.0, ramp_s=0.3).target_pos
        rest = runner.run(seconds=1.5).target_pos
        self.assertAlmostEqual(rest[1], back[1], delta=0.002)  # neutral again: stops
        farther = runner.run(offset=(0.0, -0.08, 0.0), seconds=1.5, ramp_s=0.3).target_pos
        self.assertLess(farther[1], rest[1] - 0.02)  # farther -> comes back
        np.testing.assert_allclose(farther[[0, 2]], 0.0, atol=0.004)

    def test_still_hand_does_not_move_the_target(self):
        runner = _Runner(noise_px=2.0, world_noise_m=0.004, seed=3)
        runner.run(seconds=2.0)
        targets, angles = [], []
        for _ in range(8):
            status = runner.run(seconds=1.0)
            targets.append(status.target_pos)
            angles.append(np.linalg.norm(matrix_to_rotvec(status.target_ori)))
        self.assertLess(float(np.ptp(np.array(targets), axis=0).max()), 0.002)
        self.assertLess(float(np.degrees(np.ptp(angles))), 1.0)

    def test_single_bad_frame_is_ignored(self):
        runner = _Runner()
        before = runner.run(seconds=2.0).target_pos

        def glitch(detection, k):
            if k == 5:
                detection.image_landmarks[:, 0] = np.clip(detection.image_landmarks[:, 0] - 0.25, 0.05, 0.95)
            return detection

        after = runner.run(seconds=1.0, frame=glitch).target_pos
        np.testing.assert_allclose(after, before, atol=0.003)

    def test_lost_hand_holds_and_reacquire_does_not_jump(self):
        runner = _Runner()
        runner.run(seconds=2.0)
        runner.robot_pos = np.array((0.3, 0.1, 0.2))  # where the robot is now

        def gone(detection, k):
            detection.image_landmarks[:, 0] = -1.0  # out of frame -> rejected
            return detection

        self.assertEqual(runner.run(seconds=0.6, frame=gone).state, "lost")
        runner.offset = np.array((0.10, 0.0, 0.0))  # hand comes back elsewhere
        status = runner.run(seconds=2.0)
        self.assertEqual(status.state, "tracking")
        np.testing.assert_allclose(status.target_pos, runner.robot_pos, atol=0.003)

    def test_pause_holds_and_resume_continues_from_the_robot(self):
        runner = _Runner()
        runner.run(seconds=2.0)
        self.assertTrue(runner.mapper.toggle_pause())
        self.assertEqual(runner.run(offset=(0.1, 0, 0), seconds=1.0).state, "paused")
        runner.robot_pos = np.array((0.05, 0.0, 0.0))
        self.assertFalse(runner.mapper.toggle_pause())
        status = runner.run(seconds=0.5)
        np.testing.assert_allclose(status.target_pos, runner.robot_pos, atol=0.003)

    def test_whole_palm_rotation_is_followed(self):
        runner = _Runner()
        runner.run(seconds=2.0)
        # Roll about your fingers -> robot turns about its finger axis (EE z).
        roll = np.degrees(matrix_to_rotvec(runner.run(rotdeg=(15, 0, 0), seconds=2.0).target_ori))
        np.testing.assert_allclose(roll, (0, 0, 15), atol=4.0)
        runner = _Runner()
        runner.run(seconds=2.0)
        rotvec = np.degrees(matrix_to_rotvec(runner.run(rotdeg=(0, 20, 30), seconds=2.0).target_ori))
        self.assertGreater(abs(rotvec[0]), 12.0)  # fingers tipped toward the camera -> robot tilt
        self.assertGreater(abs(rotvec[1]), 20.0)  # knob turn -> robot yaw

    def test_knob_turn_yaws_and_rotation_is_held_while_translating(self):
        runner = _Runner()
        runner.run(seconds=2.0)
        status = runner.run(rotdeg=(0, 0, 30), seconds=2.0)
        np.testing.assert_allclose(np.degrees(matrix_to_rotvec(status.target_ori)), (0, 30, 0), atol=3.0)
        np.testing.assert_allclose(status.target_pos, 0.0, atol=0.003)  # rotation alone: no shift
        moved = runner.run(offset=(0.10, 0.0, 0.0), rotdeg=(0, 0, 10), seconds=1.5, ramp_s=0.5)
        self.assertGreater(moved.target_pos[0], 0.05)
        np.testing.assert_allclose(np.degrees(matrix_to_rotvec(moved.target_ori)), (0, 30, 0), atol=3.0)

    def test_refused_motion_is_not_wound_up(self):
        runner = _Runner()
        runner.run(seconds=2.0)
        runner.run(offset=(0.10, 0.0, 0.0), seconds=1.0, ramp_s=0.5)
        runner.mapper.sync_target(np.zeros(3), np.eye(3))  # e.g. a workspace wall at x = 0
        back = runner.run(offset=(0.05, 0.0, 0.0), seconds=1.0, ramp_s=0.5).target_pos
        self.assertLess(back[0], -0.02)  # moving back moves the arm at once

    def test_left_hand_never_moves_the_arm(self):
        class Fingers:
            def update_world_landmarks(self, *args):
                return "finger-command"

        mapper = CameraArmMapper()
        adapter = CameraArmHandInput(Fingers(), mapper, camera_mirror=True)
        detection = synthesize(START, np.eye(3))
        for k in range(60):
            result = adapter.update_hand_frame(detection.image_landmarks, detection.world_landmarks, "Left",
                                               k / 30.0, detection.frame_shape)
        self.assertEqual(result, "finger-command")
        self.assertEqual(mapper.target(np.zeros(3), np.eye(3), np.eye(3), 2.0).state, "waiting")

    def test_left_hand_next_to_the_right_one_is_a_clutch(self):
        class Fingers:
            def update_world_landmarks(self, *args):
                return None

        mapper = CameraArmMapper()
        adapter = CameraArmHandInput(Fingers(), mapper, camera_mirror=True)
        rng = np.random.default_rng(5)
        clock = {"t": 0.0, "robot": np.zeros(3), "hand": np.zeros(3)}

        def run(offset, seconds, hands=("Right",)):
            begin, end = clock["hand"], np.asarray(offset, dtype=float)
            status = None
            for k in range(int(seconds * 30)):
                frac = min(1.0, (k + 1) / 15.0)
                d = synthesize(START + control_offset_to_camera(begin + frac * (end - begin)), np.eye(3),
                               image_noise=1 / 640, world_noise_m=0.002, rng=rng)
                adapter.update_hand_frame(d.image_landmarks, d.world_landmarks, "Right", clock["t"], d.frame_shape,
                                          detected_hands=hands)
                status = mapper.target(clock["robot"], np.eye(3), np.eye(3), clock["t"])
                clock["t"] += 1 / 30
            clock["hand"] = end
            return status

        run((0, 0, 0), 2.0)
        self.assertEqual(run((0.06, 0, 0), 1.0).state, "tracking")
        self.assertEqual(run((0.06, 0, 0), 0.2, hands=("Right", "Left")).state, "clutched")
        held = run((-0.04, 0, 0), 1.0, hands=("Right", "Left"))  # reposition: the arm holds
        self.assertEqual(held.state, "clutched")
        clock["robot"] = np.array((0.07, 0.0, 0.0))  # where the arm stopped
        resumed = run((-0.04, 0, 0), 1.0)
        self.assertEqual(resumed.state, "tracking")
        np.testing.assert_allclose(resumed.target_pos, clock["robot"], atol=0.003)  # no jump
        again = run((0.02, 0, 0), 1.5)
        self.assertGreater(again.target_pos[0], clock["robot"][0] + 0.02)  # follows again from there


class SettingsTests(unittest.TestCase):
    def test_gain_curve_and_depth_rate(self):
        s = CameraArmSettings()
        self.assertAlmostEqual(s.speed_gain(0.0), s.slow_gain)
        self.assertAlmostEqual(s.speed_gain(1.0), s.fast_gain)
        speeds = np.linspace(0.0, 0.2, 21)
        self.assertTrue(np.all(np.diff([s.speed_gain(v) for v in speeds]) >= 0.0))
        self.assertEqual(s.depth_rate(0.5 * s.depth_deadzone_m), 0.0)
        self.assertGreater(s.depth_rate(s.depth_deadzone_m + 0.02), 0.0)
        self.assertLess(s.depth_rate(-s.depth_deadzone_m - 0.02), 0.0)
        self.assertAlmostEqual(s.depth_rate(1.0), s.depth_max_speed_m_s)


class OperatorViewTests(unittest.TestCase):
    def test_plan_and_elevation_views_follow_the_operator_basis(self):
        from wrist_teleop.operator_views import horizontal_axes, plan_and_elevation_views

        angle = np.radians(35.0)
        basis = np.array(((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0, 0, 1.0)))
        right, away = horizontal_axes(basis)
        points = np.array(((0.0, 0.0), (0.3, 0.2), (-0.2, 0.3)))
        top, side = plan_and_elevation_views(points, (0.9, 1.5), basis, top_camera="top", top_size=(640, 535),
                                             side_camera="side", side_size=(640, 535), side_sign=-1.0)
        centre = np.array((*top.position[:2], 1.0))
        u0, v0 = top.project(centre)
        self.assertEqual((u0, v0), (320, 268))
        u, v = top.project(centre + 0.1 * np.append(right[:2], 0.0))
        self.assertGreater(u, u0)  # your right -> image right
        u, v = top.project(centre + 0.1 * np.append(away[:2], 0.0))
        self.assertLess(v, v0)  # away -> image up
        s0 = side.project(np.array((0.0, 0.0, 1.0)))
        self.assertLess(side.project(np.array((0.0, 0.0, 1.1)))[1], s0[1])  # up -> image up
        for point in points:  # every object is inside both views
            for view in (top, side):
                u, v = view.project(np.array((*point, 1.0)))
                self.assertTrue(0 <= u < view.size_px[0] and 0 <= v < view.size_px[1])
        np.testing.assert_allclose(np.cross(top.x_axis, top.y_axis), (0, 0, 1), atol=1e-9)  # looks down


class FollowerAndLimitTests(unittest.TestCase):
    def test_follower_caps_speed_and_settles(self):
        follower = TargetFollower(max_speed_m_s=0.2, max_rot_speed_rad_s=1.0)
        step = follower.delta((1.0, 0, 0), rotvec_to_matrix((0, 0, 1.0)), np.zeros(3), np.eye(3), 0.05)
        self.assertAlmostEqual(float(np.linalg.norm(step[:3])), 0.01)
        self.assertAlmostEqual(float(np.linalg.norm(step[3:])), 0.05)
        np.testing.assert_array_equal(follower.delta((0.001, 0, 0), np.eye(3), np.zeros(3), np.eye(3), 0.05), np.zeros(6))

    def test_workspace_box_reach_base_and_orientation(self):
        limits = WorkspaceLimits(box_min=np.array((-0.5, -0.5, 0.9)), box_max=np.array((0.5, 0.5, 1.4)),
                                 shoulder=np.array((-0.6, 0.0, 1.25)), base_axis_xy=np.array((-0.6, 0.0)),
                                 max_reach_m=0.8, min_radius_m=0.3, max_yaw_rad=2.0, max_tilt_rad=1.0)
        position, _, reasons = limits.clamp(np.array((0.0, 0.0, 0.5)), np.eye(3))
        self.assertEqual(position[2], 0.9)
        self.assertIn("workspace_box", reasons)
        position, _, reasons = limits.clamp(np.array((0.5, 0.0, 1.25)), np.eye(3))
        self.assertAlmostEqual(float(np.linalg.norm(position - limits.shoulder)), 0.8)
        self.assertIn("reach_limit", reasons)
        position, _, reasons = limits.clamp(np.array((-0.5, 0.0, 1.0)), np.eye(3))
        self.assertAlmostEqual(float(np.linalg.norm(position[:2] - limits.base_axis_xy)), 0.3)
        self.assertIn("base_axis_limit", reasons)
        _, orientation, reasons = limits.clamp(np.array((0.0, 0.0, 1.0)), rotvec_to_matrix((0, 0, 2.8)))
        self.assertAlmostEqual(float(np.linalg.norm(matrix_to_rotvec(orientation))), 2.0, places=5)
        _, orientation, reasons = limits.clamp(np.array((0.0, 0.0, 1.0)), rotvec_to_matrix((1.4, 0, 0)))
        self.assertAlmostEqual(float(np.linalg.norm(matrix_to_rotvec(orientation))), 1.0, places=5)
        self.assertIn("orientation_limit", reasons)


if __name__ == "__main__":
    unittest.main()
