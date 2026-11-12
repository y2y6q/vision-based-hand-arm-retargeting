"""Approach planner entry: geometry, routes, paths, takeover, slow motion, worker, mount switch.

The geometry / path / state-machine tests run in any interpreter; the scene
tests need .venv-libero (robosuite 1.4), the worker tests also the
``ompl_planner`` Conda env:

    .\\.venv-libero\\Scripts\\python.exe -m pytest -q .\\tests\\test_libero_planner_teleop.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.libero_motion_planning import (  # noqa: E402
    ApproachPlanner,
    HandState,
    PlannedMotion,
    SceneTarget,
    TargetBox,
    approach_goal,
    cartesian_candidates,
    diagonal_waypoints,
    discover_planner_python,
    finger_root_center,
    grasp_center,
    mat_to_quat,
    palm_faces_down,
    quat_to_mat,
    rotation_angle,
    straight_waypoints,
)

try:
    import robosuite

    LEGACY_ROBOSUITE = robosuite.__version__.startswith("1.4")
except ImportError:  # pragma: no cover
    LEGACY_ROBOSUITE = False


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hand(ee=(0.0, 0.0, 0.3), held=None) -> HandState:
    ee = np.asarray(ee, dtype=np.float64)
    tips = {"th": ee + (0.03, 0.0, -0.10), "ff": ee + (-0.03, 0.01, -0.12), "mf": ee + (-0.03, -0.01, -0.12),
            "rf": ee + (-0.03, -0.03, -0.11), "lf": ee + (-0.03, -0.05, -0.10)}
    return HandState(ee, np.eye(3), tips, held)


class GeometryTests(unittest.TestCase):
    def test_grasp_center_is_between_thumb_and_index_middle(self):
        hand = _hand()
        center = grasp_center(hand.tips)
        np.testing.assert_allclose(center, hand.ee_pos + (0.0, 0.0, -0.11))

    def test_approach_goal_aligns_grasp_space_and_clears_the_top(self):
        hand = _hand()
        box = TargetBox(np.array([0.2, 0.1, 0.0]), np.array([0.26, 0.15, 0.08]))
        goal = approach_goal(hand, box, clearance_m=0.02)
        shifted = HandState(goal, hand.ee_rot, {k: v + (goal - hand.ee_pos) for k, v in hand.tips.items()})
        np.testing.assert_allclose(grasp_center(shifted.tips)[:2], box.center[:2], atol=1e-12)
        self.assertAlmostEqual(shifted.lowest_z, 0.08 + 0.02, places=12)

    def test_palm_down_aims_the_finger_roots_else_the_grasp_centre(self):
        hand = _hand()
        roots = {"ff": hand.ee_pos + (0.06, 0.02, -0.01), "mf": hand.ee_pos + (0.06, 0.0, -0.01),
                 "rf": hand.ee_pos + (0.06, -0.02, -0.01)}
        down = HandState(hand.ee_pos, hand.ee_rot, hand.tips, roots=roots, palm_normal=np.array([0.1, 0.0, -1.0]))
        self.assertEqual(down.aim, "finger roots")
        np.testing.assert_allclose(down.reference_point, hand.ee_pos + (0.06, 0.0, -0.01))
        box = TargetBox(np.array([0.2, 0.1, 0.0]), np.array([0.26, 0.15, 0.08]))
        shift = approach_goal(down, box, clearance_m=0.02) - down.ee_pos
        np.testing.assert_allclose((finger_root_center(roots) + shift)[:2], box.center[:2], atol=1e-12)
        self.assertAlmostEqual(down.lowest_z + shift[2], 0.08 + 0.02, places=12)  # height rule unchanged
        # Palm sideways (the down mount: fingers down), or no roots: the grasp centre as before.
        side = HandState(hand.ee_pos, hand.ee_rot, hand.tips, roots=roots, palm_normal=np.array([-1.0, 0.0, 0.1]))
        self.assertEqual(side.aim, "grasp centre")
        np.testing.assert_allclose(side.reference_point, grasp_center(hand.tips))
        self.assertEqual(hand.aim, "grasp centre")
        self.assertTrue(palm_faces_down(np.array([0.0, 0.69, -0.72])))  # 44 deg from straight down
        self.assertFalse(palm_faces_down(np.array([0.0, 0.73, -0.68])))  # 47 deg
        self.assertFalse(palm_faces_down(None))
        # Carrying something: the held object is aligned, whatever the palm does.
        held = TargetBox(np.array([-0.02, -0.02, 0.12]), np.array([0.02, 0.02, 0.19]))
        carrying = HandState(hand.ee_pos, hand.ee_rot, hand.tips, held, roots=roots,
                             palm_normal=np.array([0.0, 0.0, -1.0]))
        self.assertEqual(carrying.aim, "held object")
        np.testing.assert_allclose(carrying.reference_point, held.center)

    def test_held_object_is_aligned_and_cleared_instead_of_the_fingers(self):
        held = TargetBox(np.array([-0.02, -0.02, 0.12]), np.array([0.02, 0.02, 0.19]))
        hand = _hand(held=held)
        basket = TargetBox(np.array([0.3, 0.3, 0.0]), np.array([0.5, 0.5, 0.10]))
        goal = approach_goal(hand, basket, clearance_m=0.02)
        shift = goal - hand.ee_pos
        np.testing.assert_allclose((held.center + shift)[:2], basket.center[:2], atol=1e-12)
        self.assertAlmostEqual(held.lower[2] + shift[2], 0.12, places=12)  # held bottom 2 cm over the rim

    def test_straight_path_lifts_over_what_it_crosses(self):
        hand = _hand(ee=(0.0, 0.0, 0.2))  # lowest tip at 0.08
        tall = TargetBox(np.array([0.15, -0.05, 0.0]), np.array([0.25, 0.05, 0.20]))  # in the way
        aside = TargetBox(np.array([0.15, 0.5, 0.0]), np.array([0.25, 0.6, 0.90]))  # not crossed
        goal = np.array([0.4, 0.0, 0.18])
        points = straight_waypoints(hand, goal, [tall, aside], margin_m=0.05)
        self.assertEqual(len(points), 4)
        np.testing.assert_allclose(points[0], hand.ee_pos)
        np.testing.assert_allclose(points[-1], goal)
        lift = points[1][2] - hand.ee_pos[2]
        self.assertAlmostEqual(hand.lowest_z + lift, 0.25, places=12)  # tall top + margin, not the aside box
        self.assertEqual(points[1][2], points[2][2])

    def test_diagonal_route_sinks_to_the_clearing_height_from_a_high_start(self):
        hand = _hand(ee=(0.0, 0.0, 0.40))  # lowest tip at 0.28
        low = TargetBox(np.array([0.35, -0.05, 0.0]), np.array([0.45, 0.05, 0.05]))  # the target
        goal = approach_goal(hand, low, 0.02)
        high = straight_waypoints(hand, goal, [low], margin_m=0.05)
        self.assertAlmostEqual(high[1][2], 0.40, places=12)  # never below the start
        diagonal = diagonal_waypoints(hand, goal, [low], margin_m=0.05)
        self.assertEqual(len(diagonal), 3)
        np.testing.assert_allclose(diagonal[1][:2], goal[:2])
        self.assertAlmostEqual(hand.lowest_z + diagonal[1][2] - hand.ee_pos[2], 0.05 + 0.05, places=12)
        np.testing.assert_allclose(diagonal[-1], goal)

    def test_candidates_order_and_deduplication(self):
        hand = _hand(ee=(0.0, 0.0, 0.40))
        target = TargetBox(np.array([0.35, -0.05, 0.0]), np.array([0.45, 0.05, 0.05]))
        goal = approach_goal(hand, target, 0.02)
        routes = cartesian_candidates(hand, goal, [target])
        # Start high: the three lift margins give one lift / traverse / descend
        # route; then the diagonals at three heights.
        self.assertEqual(len(routes), 4)
        np.testing.assert_allclose(routes[0][1][2], 0.40)
        self.assertEqual([len(r) for r in routes[1:]], [3, 3, 3])
        heights = [r[1][2] for r in routes[1:]]
        self.assertTrue(heights[0] < heights[1] < heights[2])
        for route in routes:
            np.testing.assert_allclose(route[0], hand.ee_pos)
            np.testing.assert_allclose(route[-1], goal)
        # A goal within 5 cm: the direct line comes first.
        near = cartesian_candidates(hand, hand.ee_pos + (0.03, 0.0, 0.0), [target])
        self.assertEqual(len(near[0]), 2)

    def test_quaternion_round_trip_and_angle(self):
        rotation = _rot_z(0.7) @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(quat_to_mat(mat_to_quat(rotation)), rotation, atol=1e-12)
        self.assertAlmostEqual(rotation_angle(np.eye(3), _rot_z(0.3)), 0.3, places=9)


class MotionTests(unittest.TestCase):
    def test_min_jerk_path_starts_and_ends_at_rest_within_speed_limits(self):
        positions = [np.zeros(3), np.array([0.0, 0.0, 0.1]), np.array([0.3, 0.0, 0.1])]
        rotations = [np.eye(3), np.eye(3), _rot_z(0.4)]
        motion = PlannedMotion(positions, rotations, max_speed_m_s=0.15, max_angular_speed_rad_s=0.8)
        dt = 0.002
        times = np.arange(0.0, motion.duration_s + dt, dt)
        samples = [motion.sample(t) for t in times]
        points = np.array([p for p, _r in samples])
        speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
        self.assertLessEqual(float(speeds.max()), 0.15 * 1.001)
        self.assertLess(float(speeds[:3].max()), 0.002)  # starts at rest
        self.assertLess(float(speeds[-3:].max()), 0.002)  # ends at rest
        np.testing.assert_allclose(points[-1], positions[-1], atol=1e-12)
        self.assertAlmostEqual(rotation_angle(samples[-1][1], _rot_z(0.4)), 0.0, places=6)
        angular = [rotation_angle(a[1], b[1]) / dt for a, b in zip(samples, samples[1:])]
        self.assertLessEqual(max(angular), 0.8 * 1.01)

    def test_segmented_path_rests_at_each_corner_and_stays_on_the_lines(self):
        positions = [np.zeros(3), np.array([0.0, 0.0, 0.1]), np.array([0.3, 0.0, 0.1]), np.array([0.3, 0.0, 0.02])]
        rotations = [np.eye(3)] * 4
        single = PlannedMotion(positions, rotations, max_speed_m_s=0.3)
        motion = PlannedMotion(positions, rotations, max_speed_m_s=0.3, segmented=True)
        self.assertAlmostEqual(motion.duration_s, single.duration_s, places=12)  # same total time
        dt = 0.001
        times = np.arange(0.0, motion.duration_s + dt, dt)
        points = np.array([motion.sample(t)[0] for t in times])
        speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
        self.assertLessEqual(float(speeds.max()), 0.3 * 1.001)
        for corner_time in motion._starts[1:-1]:  # at rest at the inner corners
            index = int(round(corner_time / dt))
            self.assertLess(float(speeds[max(0, index - 1):index + 1].max()), 0.002)
        # Every sample lies on one of the three straight segments.
        for point in points[::25]:
            on_lift = abs(point[0]) < 1e-9 and abs(point[1]) < 1e-9
            on_traverse = abs(point[2] - 0.1) < 1e-9 and abs(point[1]) < 1e-9
            on_descent = abs(point[0] - 0.3) < 1e-9 and abs(point[1]) < 1e-9
            self.assertTrue(on_lift or on_traverse or on_descent, point)
        np.testing.assert_allclose(points[-1], positions[-1], atol=1e-12)

    def test_planner_hands_back_on_arrival_and_on_cancel(self):
        plan = lambda request: ([request.goal_pos], [request.goal_rot], {"backend": "fake"})  # noqa: E731
        target = SceneTarget(1, "bowl", "object")
        planner = ApproachPlanner(plan, "fake")
        hand = _hand()
        goal = hand.ee_pos + (0.05, 0.0, 0.0)
        planner.start(target, goal, np.eye(3), hand, [])
        self.assertEqual(planner.state, "planning")
        deadline = time.monotonic() + 2.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = planner.tick(0.05, hand.ee_pos, np.eye(3))
            time.sleep(0.01)
        self.assertEqual(planner.state, "moving")
        ee = hand.ee_pos.copy()
        for _ in range(200):  # a perfect tracker
            result = planner.tick(0.05, ee, np.eye(3))
            if result is None:
                break
            ee = result[0]
        self.assertEqual(planner.state, "idle")
        events = [event["event"] for event in planner.drain_events()]
        self.assertEqual(events, ["planner_request", "planner_result", "planner_arrived"])
        np.testing.assert_allclose(ee, goal, atol=1e-9)
        # Cancel while planning: a late result is ignored and the arm stays manual.
        slow = ApproachPlanner(lambda request: (time.sleep(0.2), plan(request))[1], "fake")
        slow.start(target, goal, np.eye(3), hand, [])
        self.assertTrue(slow.cancel("spacemouse"))
        time.sleep(0.3)
        self.assertIsNone(slow.tick(0.05, hand.ee_pos, np.eye(3)))
        self.assertEqual(slow.state, "idle")
        self.assertFalse(slow.cancel("again"))


class SimSpeedTests(unittest.TestCase):
    def test_sim_speed_sets_control_frequency_and_scales_increments(self):
        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_planner_teleop import arm_config_for_speeds, env_config_for_sim_speed

        config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
        for speed, hz in ((1.0, 20.0), (0.8, 25.0), (0.6, 20.0 / 0.6), (0.4, 50.0)):
            self.assertAlmostEqual(env_config_for_sim_speed(config, speed).control_hz, hz)
            # Each tick: a whole number of 2 ms physics steps.
            substeps = int((1.0 / round(hz)) / 0.002)
            self.assertAlmostEqual(substeps * 0.002, 0.05 * speed, places=9)
        with self.assertRaises(ValueError):
            env_config_for_sim_speed(config, 0.5)
        slow = arm_config_for_speeds(config, 3.0, 2.0, 0.6)
        self.assertAlmostEqual(slow.max_translation_delta_m, config.max_translation_delta_m * 3.0 * 0.6)
        self.assertAlmostEqual(slow.max_rotation_delta_rad, config.max_rotation_delta_rad * 2.0 * 0.6)
        self.assertEqual(slow.control_hz, config.control_hz)

    def test_planner_speed_scales_both_peaks(self):
        fast = ApproachPlanner(lambda request: None, "fake", max_speed_m_s=0.4)
        self.assertAlmostEqual(fast.max_angular_speed_rad_s, 0.4 * 0.8 / 0.15)


class ViewAndMountTests(unittest.TestCase):
    def test_projection_matches_a_pinhole_camera(self):
        from wrist_teleop.libero_planner_teleop import project_point

        # Camera at the origin looking along +x, up +z; 90 deg vertical field of view.
        glcam = SimpleNamespace(pos=(0.0, 0.0, 0.0), forward=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
                                frustum_near=0.1, frustum_top=0.1, frustum_bottom=-0.1, frustum_width=0.0,
                                frustum_center=0.0)
        width, height = 800, 400
        u, v = project_point(glcam, (2.0, 0.0, 0.0), width, height)
        self.assertAlmostEqual(u, 400.0)
        self.assertAlmostEqual(v, 200.0)
        _u, v = project_point(glcam, (2.0, 0.0, 2.0), width, height)  # 45 deg up: the top edge
        self.assertAlmostEqual(v, 0.0)
        # right = forward x up = -y; half-width from the 2:1 aspect is 0.2 at the near plane.
        u, _v = project_point(glcam, (1.0, -1.0, 0.0), width, height)
        self.assertAlmostEqual(u, 400.0 + 0.5 * 800.0 * (0.1 / 0.2))
        self.assertIsNone(project_point(glcam, (-1.0, 0.0, 0.0), width, height))  # behind

    def test_status_text_carries_badges(self):
        from wrist_teleop.libero_planner_teleop import _AnnotatedStatus

        status = _AnnotatedStatus("LIBERO", [(10, 20, "3")])
        self.assertEqual(str(status), "LIBERO")
        self.assertEqual(status.badges, ((10, 20, "3"),))
        self.assertFalse(_AnnotatedStatus(""))

    def test_mount_helpers(self):
        from wrist_teleop.libero_planner_teleop import MOUNT_SWITCH_KEY, mount_of, other_mount
        from wrist_teleop.libero_spatial_teleop import shadow_spec

        self.assertEqual(mount_of(shadow_spec("down")), "down")
        self.assertEqual(mount_of(shadow_spec("forward")), "forward")
        self.assertEqual(other_mount("down"), "forward")
        self.assertEqual(other_mount("forward"), "down")
        self.assertEqual(MOUNT_SWITCH_KEY, "m")  # not C: the camera inset prints "C re-centre dex"


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class LiberoPlannerSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from wrist_teleop.config import WristConfig
        from wrist_teleop import libero_planner_teleop as P
        from wrist_teleop import libero_spatial_teleop as L

        cls.P, cls.L = P, L
        cls.config = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")

    def _scene(self, suite="libero_object", task_id=0, sim_speed=1.0, mount="down"):
        spec = self.L.shadow_spec(mount)
        session = self.L.OfficialLiberoSpatialSession(
            spec, self.P.env_config_for_sim_speed(self.config, sim_speed), task_id=task_id, task_order=0,
            init_state_id=0, seed=7, has_offscreen_renderer=False, suite=suite)
        session.open()
        self.addCleanup(session.close)
        composer = self.P.PlannerActionComposer(session.env, spec)
        composer.initialize_goal()
        return session, composer, self.P.PlannerScene(session.env, composer)

    def _run_until_idle(self, env, composer, planner, dt, ticks=400):
        for _ in range(ticks):
            ee_pos, ee_rot = composer.ee_pose()
            goal = planner.tick(dt, ee_pos, ee_rot)
            if goal is not None:
                composer.set_goal(*goal)
            env.step(composer.compose(np.zeros(6), None))
            if planner.state == "idle":
                return True
            time.sleep(0.001)
        return False

    def test_targets_fingertips_and_rays(self):
        session, composer, scene = self._scene("libero_spatial", 0)
        names = [t.name for t in scene.targets]
        self.assertEqual(names[:5], list(session.env.objects_dict.keys()))
        self.assertIn("wooden_cabinet_1", names)
        self.assertLessEqual(len(names), 9)
        data = session.env.sim.data._data
        tips = scene.fingertips.tips(data)
        model = session.env.sim.model._model
        import mujoco
        for finger, tip in tips.items():  # the tip is beyond the distal joint along the finger
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_rh_{}distal".format(finger))
            self.assertGreater(float(np.linalg.norm(tip - data.xpos[body])), 0.025)
        # Fingers point down at the start pose: the tips are the lowest hand points.
        self.assertLess(max(t[2] for t in tips.values()), composer.ee_pose()[0][2])
        rays = self.P.FingertipRays(session.env, scene.robot_geoms)
        segments = rays.segments()
        self.assertEqual(sorted(segments), sorted(tips))
        for finger, (tip, hit) in segments.items():
            self.assertIsNotNone(hit)
            np.testing.assert_allclose(hit[:2], tip[:2], atol=1e-9)  # straight down
            self.assertLess(hit[2], tip[2])
        self.assertEqual(len(rays.geoms()), 10)
        rays.enabled = False
        self.assertEqual(rays.geoms(), [])
        badges = scene.badge_points()
        self.assertEqual([text for text, _point in badges], [str(t.number) for t in scene.targets])
        first = scene.box(scene.targets[0])
        np.testing.assert_allclose(badges[0][1], [first.center[0], first.center[1], first.top + self.P.LABEL_LIFT_M])

    def test_badges_are_projected_into_the_front_and_3d_views(self):
        session, composer, scene = self._scene("libero_object", 0)
        env = session.env
        tri = self.P.follow_robot_base(self.P._shadow_tri_view_config(self.config), env)
        window = self.P._planner_view_window(
            tri, env, composer, headless=True, view_size=(960, 540), key_handler=None, on_close=None,
            preview=None, scene_extras=lambda: [], badge_points=scene.badge_points)
        self.addCleanup(window.close)
        window.update(env, status_text="test")
        status = next(s.status_text for s in window._sets if s.status_text is not None)
        self.assertEqual(str(status), "test")
        rects = window.panel_rects()
        slots = {}
        for x, y, text in status.badges:
            slot = next(name for name, (rx, ry, rw, rh) in rects.items() if rx <= x < rx + rw and ry <= y < ry + rh)
            slots.setdefault(slot, []).append(text)
        self.assertEqual(sorted(slots), ["front", "side"])  # not the close-up
        self.assertEqual(sorted(slots["front"]), sorted(str(t.number) for t in scene.targets))

    def test_straight_approach_puts_the_grasp_space_over_the_object(self):
        from wrist_teleop.libero_motion_planning import grasp_center
        session, composer, scene = self._scene("libero_object", 0)
        env = session.env
        planner = ApproachPlanner(self.P._straight_plan, "straight")
        target = scene.by_number(1)
        hand = scene.hand_state([])
        self.assertEqual(hand.aim, "grasp centre")  # down mount: fingers down, palm sideways
        box = scene.box(target)
        goal = approach_goal(hand, box, 0.02)
        waypoints = straight_waypoints(hand, goal, [scene.box(t) for t in scene.targets])
        planner.start(target, goal, hand.ee_rot, hand, [], {"waypoints": waypoints})
        self.assertTrue(self._run_until_idle(env, composer, planner, env.control_timestep))
        events = planner.drain_events()
        self.assertEqual(events[-1]["event"], "planner_arrived")
        for _ in range(10):  # let it settle under the held goal
            env.step(composer.compose(np.zeros(6), None))
        tips = scene.fingertips.tips(env.sim.data._data)
        center = grasp_center(tips)
        self.assertLess(float(np.linalg.norm(center[:2] - box.center[:2])), 0.005)
        lowest = min(t[2] for t in tips.values())
        self.assertAlmostEqual(lowest - box.top, 0.02, delta=0.005)

    def test_forward_mount_approach_puts_the_finger_roots_over_the_object(self):
        from wrist_teleop.libero_motion_planning import finger_root_center, grasp_center
        session, composer, scene = self._scene("libero_object", 0, mount="forward")
        env = session.env
        planner = ApproachPlanner(self.P._straight_plan, "straight")
        target = scene.by_number(1)
        hand = scene.hand_state([])
        self.assertEqual(hand.aim, "finger roots")  # palm parallel to the table
        box = scene.box(target)
        goal = approach_goal(hand, box, 0.02)
        waypoints = straight_waypoints(hand, goal, [scene.box(t) for t in scene.targets])
        planner.start(target, goal, hand.ee_rot, hand, [], {"waypoints": waypoints})
        self.assertTrue(self._run_until_idle(env, composer, planner, env.control_timestep))
        events = planner.drain_events()
        self.assertEqual(events[-1]["event"], "planner_arrived")
        self.assertEqual(next(e for e in events if e["event"] == "planner_request")["aim"], "finger roots")
        for _ in range(10):  # let it settle under the held goal
            env.step(composer.compose(np.zeros(6), None))
        data = env.sim.data._data
        roots = finger_root_center(scene.finger_roots.roots(data))
        self.assertLess(float(np.linalg.norm(roots[:2] - box.center[:2])), 0.005)
        tips = scene.fingertips.tips(data)
        self.assertAlmostEqual(min(t[2] for t in tips.values()) - box.top, 0.02, delta=0.005)
        # The old aim point (the grasp centre) would have left the object under the fingertips.
        self.assertGreater(float(np.linalg.norm(grasp_center(tips)[:2] - box.center[:2])), 0.04)

    def test_takeover_mid_move_stops_the_arm_on_the_spot(self):
        session, composer, scene = self._scene("libero_object", 0)
        env = session.env
        planner = ApproachPlanner(self.P._straight_plan, "straight")
        target = scene.by_number(2)  # basket: a long move
        hand = scene.hand_state([])
        goal = approach_goal(hand, scene.box(target), 0.02)
        planner.start(target, goal, hand.ee_rot, hand, [],
                      {"waypoints": straight_waypoints(hand, goal, [scene.box(t) for t in scene.targets])})
        moved = 0
        for _ in range(200):  # until it is moving fast
            ee_pos, ee_rot = composer.ee_pose()
            goal_now = planner.tick(env.control_timestep, ee_pos, ee_rot)
            if goal_now is not None:
                composer.set_goal(*goal_now)
                moved += 1
            env.step(composer.compose(np.zeros(6), None))
            if moved >= 30:
                break
            time.sleep(0.001)
        self.assertEqual(planner.state, "moving")
        # What the loop does on SpaceMouse input.
        self.assertTrue(planner.cancel("spacemouse"))
        composer.hold_here()
        stop = composer.ee_pose()[0]
        for _ in range(20):  # 1 s with zero input
            ee_pos, ee_rot = composer.ee_pose()
            self.assertIsNone(planner.tick(env.control_timestep, ee_pos, ee_rot))
            env.step(composer.compose(np.zeros(6), None))
        after = composer.ee_pose()[0]
        self.assertLess(float(np.linalg.norm(after - stop)), 0.02)  # coasts < 2 cm, then holds
        last = composer.ee_pose()[0]
        env.step(composer.compose(np.zeros(6), None))
        self.assertLess(float(np.linalg.norm(composer.ee_pose()[0] - last)), 0.0005)
        self.assertEqual([e["event"] for e in planner.drain_events()][-1], "planner_cancel")

    def test_slow_motion_steps_less_simulated_time_per_tick(self):
        session, _composer, _scene = self._scene("libero_object", 0, sim_speed=0.6)
        env = session.env
        before = float(env.sim.data.time)
        env.step(np.zeros(env.action_dim))
        self.assertAlmostEqual(float(env.sim.data.time) - before, 0.030, places=6)

    # ------------------------------------------------ worker (ompl_planner env)
    def _worker(self, env, scene):
        """A sidecar plus the init request for this scene (the compiled model in a temp dir)."""
        import mujoco
        from wrist_teleop.libero_motion_planning import OmplPlannerSidecar

        scratch = tempfile.TemporaryDirectory(prefix="planner_test_")
        self.addCleanup(scratch.cleanup)  # runs after the sidecar closes (cleanups are LIFO)
        sidecar = OmplPlannerSidecar(discover_planner_python())
        self.addCleanup(sidecar.close)
        mjb = Path(scratch.name) / "scene.mjb"
        mujoco.mj_saveModel(env.sim.model._model, str(mjb), None)
        init = {"model_path": str(mjb), "arm_qpos_adr": scene.arm_qpos_adr, "arm_dof_adr": scene.arm_dof_adr,
                "arm_joint_ids": scene.arm_joint_ids, "eef_site": scene.eef_site,
                "robot_geoms": sorted(scene.robot_geoms)}
        return sidecar, init

    def _request(self, env, scene, number, with_routes=True, qpos=None):
        target = scene.by_number(number)
        hand = scene.hand_state([])
        goal = approach_goal(hand, scene.box(target), 0.02)
        payload = {"qpos": np.asarray(env.sim.data._data.qpos if qpos is None else qpos).tolist(),
                   "goal_pos": goal.tolist(), "goal_rot": hand.ee_rot.ravel().tolist(), "held": [],
                   "time_limit_s": 2.0, "simplify_s": 0.5, "seed": 0}
        if with_routes:
            obstacles = [scene.box(t) for t in scene.targets]
            payload["candidates"] = [[p.tolist() for p in r] for r in cartesian_candidates(hand, goal, obstacles)]
        return target, hand, goal, payload

    def _drive(self, env, composer, scene, planner, start_rot):
        """Run the move; return (deepest robot-vs-scene contact, largest hand turn from the start)."""
        robot = scene.robot_geoms
        deepest, turn = 0.0, 0.0
        for _ in range(500):
            ee_pos, ee_rot = composer.ee_pose()
            turn = max(turn, rotation_angle(start_rot, ee_rot))
            goal_now = planner.tick(env.control_timestep, ee_pos, ee_rot)
            if goal_now is not None:
                composer.set_goal(*goal_now)
            env.step(composer.compose(np.zeros(6), None))
            data = env.sim.data._data
            for index in range(data.ncon):  # robot vs scene during the move
                contact = data.contact[index]
                if (contact.geom1 in robot) != (contact.geom2 in robot):
                    deepest = min(deepest, float(contact.dist))
            if planner.state == "idle":
                break
        return deepest, turn

    def _plan_with(self, planner, env, composer, target, goal, hand, payload):
        planner.start(target, goal, hand.ee_rot, hand, [], payload)
        deadline = time.monotonic() + 30.0
        while planner.state == "planning" and time.monotonic() < deadline:
            ee_pos, ee_rot = composer.ee_pose()
            planner.tick(env.control_timestep, ee_pos, ee_rot)
            time.sleep(0.02)
        self.assertEqual(planner.state, "moving", list(planner.events))
        return [e for e in planner.events if e["event"] == "planner_result"][0]

    @unittest.skipIf(discover_planner_python() is None, "ompl_planner Conda env not found")
    def test_checked_route_keeps_the_hand_orientation_and_clearance(self):
        session, composer, scene = self._scene("libero_object", 0)
        env = session.env
        sidecar, init = self._worker(env, scene)
        planner = ApproachPlanner(self.P._OmplBridge(sidecar).plan, "ompl")
        target, hand, goal, payload = self._request(env, scene, 2)  # basket: the long carry
        result = self._plan_with(planner, env, composer, target, goal, hand,
                                 {"scene": {"version": 1, "init": init}, **payload})
        self.assertEqual(result["strategy"], "cartesian")
        self.assertTrue(result["segmented"])
        deepest, turn = self._drive(env, composer, scene, planner, hand.ee_rot)
        self.assertEqual(planner.drain_events()[-1]["event"], "planner_arrived")
        self.assertGreater(deepest, -0.003)
        self.assertLess(np.degrees(turn), 3.0)  # no swinging round: the hand keeps its orientation

    @unittest.skipIf(discover_planner_python() is None, "ompl_planner Conda env not found")
    def test_ompl_fallback_is_collision_free_and_arrives(self):
        session, composer, scene = self._scene("libero_spatial", 0)
        env = session.env
        sidecar, init = self._worker(env, scene)
        planner = ApproachPlanner(self.P._OmplBridge(sidecar).plan, "ompl")
        target, hand, goal, payload = self._request(env, scene, 5, with_routes=False)  # plate_1
        self.assertEqual(target.name, "plate_1")
        result = self._plan_with(planner, env, composer, target, goal, hand,
                                 {"scene": {"version": 1, "init": init}, **payload})
        self.assertEqual(result["strategy"], "ompl")
        self.assertGreater(result["validity_checks"], 0)
        deepest, turn = self._drive(env, composer, scene, planner, hand.ee_rot)
        self.assertEqual(planner.drain_events()[-1]["event"], "planner_arrived")
        self.assertGreater(deepest, -0.003)
        self.assertLess(turn, 0.5 + 0.1)  # the orientation tube, plus tracking
        tips = scene.fingertips.tips(env.sim.data._data)
        self.assertGreater(min(t[2] for t in tips.values()), scene.box(target).top + 0.01)

    @unittest.skipIf(discover_planner_python() is None, "ompl_planner Conda env not found")
    def test_worker_plans_from_a_joint_limit_and_rejects_a_route_through_an_object(self):
        session, _composer, scene = self._scene("libero_object", 0)
        env = session.env
        sidecar, init = self._worker(env, scene)
        sidecar.request("init", timeout_s=60.0, **init)
        import mujoco

        model, data = env.sim.model._model, env.sim.data._data
        original = np.asarray(data.qpos).copy()
        # The arm parked a little past joint 7's upper limit (the 2026-11-06 real
        # run: the old worker rejected every such start). Lift the hand 3 cm.
        joint7 = scene.arm_joint_ids[6]
        data.qpos[scene.arm_qpos_adr[6]] = float(model.jnt_range[joint7][1]) + 0.002
        mujoco.mj_forward(model, data)
        _target, hand, _goal, payload = self._request(env, scene, 1)
        up = hand.ee_pos + np.array([0.0, 0.0, 0.03])
        payload.update(goal_pos=up.tolist(), candidates=[[hand.ee_pos.tolist(), up.tolist()]])
        result = sidecar.request("plan", timeout_s=60.0, **payload)
        self.assertEqual(result["strategy"], "cartesian")
        self.assertGreater(result["start_clamped_rad"], 0.0)
        data.qpos[:] = original
        mujoco.mj_forward(model, data)
        # A route straight through the milk carton is rejected (the fallback then plans around it).
        milk = scene.by_number(5)
        self.assertEqual(milk.name, "milk_1")
        box = scene.box(milk)
        _target, hand, goal, payload = self._request(env, scene, 5)
        reach_down = hand.ee_pos[2] - hand.lowest_z  # end effector above the lowest fingertip
        high = box.top + reach_down + 0.05
        inside = box.center[2] + reach_down  # fingertips at the carton's mid-height
        through = [hand.ee_pos.tolist(), [hand.ee_pos[0], hand.ee_pos[1], high], [box.center[0], box.center[1], high],
                   [box.center[0], box.center[1], inside], goal.tolist()]
        payload["candidates"] = [through]
        result = sidecar.request("plan", timeout_s=60.0, **payload)
        self.assertEqual(result["strategy"], "ompl")
        self.assertIn("milk_1", result["route_failures"][0])


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class LiberoPlannerEntryTests(unittest.TestCase):
    def test_m_switches_the_hand_mount_and_the_planner_works_after(self):
        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_planner_teleop import run_libero_planner_teleop
        from wrist_teleop.libero_spatial_teleop import shadow_spec

        before = time.time()
        code = run_libero_planner_teleop(
            shadow_spec("down"), WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json"),
            task_id=0, task_order=0, init_state_id=0, seed=7, headless=True, duration_s=12.0, camera_index=0,
            camera_display=False, camera_mirror=True, hand_control="off", synthetic_input=True,
            suite="libero_object", planner_backend="straight", smoke_keys=[(0.5, "m"), (1.0, "1")],
        )
        self.assertEqual(code, 0)
        runs = [p for p in (ROOT / "outputs" / "libero").glob("*_panda_shadow_down_planner_teleop")
                if p.stat().st_mtime >= before - 1.0]
        self.assertTrue(runs)
        events = [json.loads(line) for line in (max(runs) / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = [e["event"] for e in events]
        switch = kinds.index("hand_mount_switch")
        self.assertEqual(events[switch]["hand_mount"], "forward")
        self.assertEqual(events[switch]["spec"], "panda_shadow")
        # R was held during the rebuild and accepted again after the first step on the new model.
        settled = [e for e in events[switch:] if e["event"] == "reset_settled"]
        self.assertEqual([e["what"] for e in settled], ["hand mount switch"])
        steps = [e for e in events[switch:] if e["event"] == "step"]
        self.assertTrue(steps and all(e["hand_mount"] == "forward" for e in steps))
        self.assertIn("planner_arrived", kinds[switch:])
        self.assertEqual(events[-1]["planner"]["mount_switches"], 1)


if __name__ == "__main__":
    unittest.main()
