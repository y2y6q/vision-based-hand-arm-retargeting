"""Focused regressions for the shared Panda / Jaco one-DoF gripper path."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.gripper_teleop import (  # noqa: E402
    GripperButtonController,
    JACO_GRIPPER_POLICY_TARGET_RATE_PER_S,
    JacoContinuousGripperButtonController,
    OneDofGripperComposer,
    ROBOT_GRIPPER_EXPECTATIONS,
    _OptionalVisualAssistance,
    create_gripper_osc_pose_env,
    reset_gripper_environment,
)
from wrist_teleop.mapper import SixDofMapper  # noqa: E402


class _FakeOpenDevice:
    def __init__(self):
        self.reset_calls = 0

    def reset_input_state(self):
        self.reset_calls += 1


class _FakeControlFrameMapper:
    def __init__(self):
        self.calls: list[tuple[object, str]] = []

    def describe_control_frame(self):
        return {"source": "fallback"}

    def set_camera_from_sim(self, env, camera_name: str):
        self.calls.append((env, camera_name))
        return {"source": "live_mujoco_cam_xmat", "primary_camera_name": camera_name}


class _FakeCompositor:
    def __init__(self):
        self.config = type("Config", (), {"front_camera_name": "frontview"})()
        self.configure_calls: list[object] = []

    def configure_cameras(self, env):
        self.configure_calls.append(env)


class OneDofGripperTeleopTests(unittest.TestCase):
    config = WristConfig(smoothing_alpha=None)

    def _env(self, robot: str):
        env = create_gripper_osc_pose_env(self.config, robot=robot, has_renderer=False)
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_actual_panda_and_jaco_build_reset_step_close_and_registered_grippers(self):
        for robot, expected_gripper in ROBOT_GRIPPER_EXPECTATIONS.items():
            with self.subTest(robot=robot):
                env = self._env(robot)
                composer = OneDofGripperComposer(env)
                composer.initialize_goal()
                detail = composer.describe()
                self.assertEqual(detail["action_dim"], env.action_dim)
                self.assertEqual(detail["arm"]["slice"], [
                    composer.arm_slice.start,
                    composer.arm_slice.stop,
                ])
                self.assertEqual(detail["gripper"]["slice"], [
                    composer.gripper_slice.start,
                    composer.gripper_slice.stop,
                ])
                self.assertEqual(detail["gripper"]["model"], expected_gripper)
                self.assertEqual(detail["gripper"]["policy_dof"], 1)
                self.assertEqual(detail["arm"]["input_type"], "delta")
                self.assertEqual(detail["arm"]["input_ref_frame"], "world")
                action = composer.compose(np.zeros(6), 0.0)
                self.assertEqual(action.shape, (env.action_dim,))
                env.step(action)

    def test_dynamic_arm_and_gripper_slices_are_isolated_for_both_models(self):
        physical = np.array([0.002, -0.001, 0.0005, 0.01, -0.02, 0.03])
        for robot in ROBOT_GRIPPER_EXPECTATIONS:
            with self.subTest(robot=robot):
                env = self._env(robot)
                composer = OneDofGripperComposer(env)
                composer.initialize_goal()
                neutral = composer.compose(np.zeros(6), 0.0)
                wrist_only = composer.compose(physical, 0.0)
                gripper_only = composer.compose(np.zeros(6), composer.gripper_high)
                np.testing.assert_allclose(wrist_only[composer.gripper_slice], 0.0)
                np.testing.assert_allclose(gripper_only[composer.arm_slice], 0.0)
                self.assertGreater(np.linalg.norm(wrist_only[composer.arm_slice]), 0.0)
                self.assertGreater(float(gripper_only[composer.gripper_slice][0]), 0.0)
                unaffected_by_wrist = np.setdiff1d(
                    np.arange(env.action_dim),
                    np.arange(composer.arm_slice.start, composer.arm_slice.stop),
                )
                unaffected_by_gripper = np.setdiff1d(
                    np.arange(env.action_dim),
                    np.arange(composer.gripper_slice.start, composer.gripper_slice.stop),
                )
                np.testing.assert_allclose(wrist_only[unaffected_by_wrist], neutral[unaffected_by_wrist])
                np.testing.assert_allclose(gripper_only[unaffected_by_gripper], neutral[unaffected_by_gripper])

    def test_optional_visual_assistance_binds_the_live_primary_camera_frame(self):
        env = object()
        mapper = _FakeControlFrameMapper()
        compositor = _FakeCompositor()
        visual = _OptionalVisualAssistance(laser=None, compositor=compositor, reason=None)
        self.assertEqual(
            visual.configure_control_frame(env, mapper),
            {"source": "live_mujoco_cam_xmat", "primary_camera_name": "frontview"},
        )
        self.assertEqual(compositor.configure_calls, [env])
        self.assertEqual(mapper.calls, [(env, "frontview")])

        disabled = _OptionalVisualAssistance(laser=None, compositor=None, reason="disabled_by_cli")
        self.assertEqual(disabled.configure_control_frame(env, mapper), {"source": "fallback"})

    def test_official_single_policy_dof_opens_and_closes_each_real_gripper(self):
        for robot in ROBOT_GRIPPER_EXPECTATIONS:
            with self.subTest(robot=robot):
                env = self._env(robot)
                composer = OneDofGripperComposer(env)
                composer.initialize_goal()
                qpos_ids = np.asarray(
                    env.robots[0]._ref_gripper_joint_pos_indexes[composer.gripper_key],
                    dtype=np.intp,
                )

                for _ in range(18):
                    env.step(composer.compose(np.zeros(6), composer.gripper_low))
                opened = env.sim.data.qpos[qpos_ids].copy()

                env.reset()
                composer = OneDofGripperComposer(env)
                composer.initialize_goal()
                for _ in range(18):
                    env.step(composer.compose(np.zeros(6), composer.gripper_high))
                closed = env.sim.data.qpos[qpos_ids].copy()
                self.assertGreater(np.linalg.norm(opened - closed), 1e-4)

    def test_button_rising_edges_and_disconnect_fail_closed(self):
        controller = GripperButtonController(self.config, open_command=-1.0, close_command=1.0)
        self.assertEqual(controller.update((False, False), valid=True).state, "hold")
        close = controller.update((True, False), valid=True)
        self.assertEqual(close.state, "close")
        self.assertEqual(close.value, 1.0)
        self.assertEqual(close.rising_edges, ("left_close",))
        held_close = controller.update((True, False), valid=True)
        self.assertEqual(held_close.state, "close")
        self.assertEqual(held_close.rising_edges, ())
        controller.update((False, False), valid=True)
        opened = controller.update((False, True), valid=True)
        self.assertEqual(opened.state, "open")
        self.assertEqual(opened.value, -1.0)
        self.assertEqual(opened.rising_edges, ("right_open",))
        safe = controller.update((False, True), valid=False)
        self.assertEqual(safe.state, "safe_zero")
        self.assertEqual(safe.value, 0.0)

        env = self._env("Panda")
        composer = OneDofGripperComposer(env)
        mapper = SixDofMapper(self.config)
        invalid_wrist = mapper.map(np.full(6, 350.0), valid=False)
        safe_action = composer.compose(invalid_wrist, safe.value)
        np.testing.assert_allclose(safe_action[composer.arm_slice], 0.0)
        np.testing.assert_allclose(safe_action[composer.gripper_slice], 0.0)

    def test_reset_clears_mapper_and_button_command_without_reopening_device(self):
        env = self._env("Panda")
        mapper = SixDofMapper(self.config)
        device = _FakeOpenDevice()
        composer = OneDofGripperComposer(env)
        buttons = GripperButtonController(
            self.config,
            open_command=composer.gripper_low,
            close_command=composer.gripper_high,
        )
        mapper.map(np.full(6, 350.0), valid=True)
        buttons.update((True, False), valid=True)
        reset_composer, action = reset_gripper_environment(
            env,
            mapper=mapper,
            device=device,
            buttons=buttons,
            keypress_callback=None,
        )
        self.assertEqual(device.reset_calls, 1)
        np.testing.assert_allclose(action[reset_composer.arm_slice], 0.0)
        np.testing.assert_allclose(action[reset_composer.gripper_slice], 0.0)
        self.assertEqual(buttons.update((False, False), valid=True).state, "hold")
        np.testing.assert_allclose(mapper.map(np.zeros(6), valid=True), 0.0)

    def test_jaco_button_short_press_long_press_release_and_actual_dt_target_rate(self):
        """Jaco uses current button levels and target integration, not edges."""

        controller = JacoContinuousGripperButtonController(
            self.config,
            open_command=-1.0,
            close_command=1.0,
        )
        # A short left press advances by rate * measured dt, then release
        # emits exact zero while retaining the target for telemetry.
        short = controller.update((True, False), valid=True, dt_s=0.04)
        self.assertEqual(short.state, "closing")
        self.assertEqual(short.rising_edges, ("left_close",))
        self.assertAlmostEqual(
            short.value,
            JACO_GRIPPER_POLICY_TARGET_RATE_PER_S * 0.04,
        )
        retained_target = controller.target
        released = controller.update((False, False), valid=True, dt_s=0.50)
        self.assertEqual(released.state, "hold")
        self.assertEqual(released.value, 0.0)
        self.assertEqual(controller.target, retained_target)
        # A held button has no additional edge but continues accumulating.
        held = controller.update((True, False), valid=True, dt_s=0.20)
        self.assertEqual(held.rising_edges, ("left_close",))
        self.assertGreater(held.value, retained_target)
        held_again = controller.update((True, False), valid=True, dt_s=0.20)
        self.assertEqual(held_again.rising_edges, ())
        self.assertGreater(held_again.value, held.value)

        # Equal elapsed time must give equal target change regardless of how
        # the control loop is split into frames.
        one_step = JacoContinuousGripperButtonController(
            self.config, open_command=-1.0, close_command=1.0
        )
        one_step.update((True, False), valid=True, dt_s=0.20)
        split_steps = JacoContinuousGripperButtonController(
            self.config, open_command=-1.0, close_command=1.0
        )
        for _ in range(4):
            split_steps.update((True, False), valid=True, dt_s=0.05)
        self.assertAlmostEqual(one_step.target, split_steps.target)

    def test_jaco_button_limits_dual_press_and_arm_slice_isolation(self):
        controller = JacoContinuousGripperButtonController(
            self.config,
            open_command=-1.0,
            close_command=1.0,
        )
        # Sustain close beyond the policy limit. Target clamps exactly and
        # does not accumulate further; output remains a legal close velocity
        # so the official SimpleGrip controller can reach its physical stop.
        for _ in range(40):
            closing = controller.update((True, False), valid=True, dt_s=0.10)
        self.assertEqual(closing.state, "close_limit")
        self.assertEqual(controller.target, 1.0)
        target_at_limit = controller.target
        for _ in range(5):
            controller.update((True, False), valid=True, dt_s=0.10)
        self.assertEqual(controller.target, target_at_limit)

        dual = controller.update((True, True), valid=True, dt_s=0.50)
        self.assertEqual(dual.state, "both_buttons_safe_hold")
        self.assertEqual(dual.value, 0.0)
        self.assertEqual(controller.target, target_at_limit)

        # Reverse directions safely, then prove the actual action composer
        # never lets the Jaco button controller overwrite OSC's six entries.
        controller.update((False, False), valid=True, dt_s=0.05)
        opening = controller.update((False, True), valid=True, dt_s=0.10)
        self.assertLess(opening.value, 0.0)
        env = self._env("Jaco")
        composer = OneDofGripperComposer(env)
        wrist = np.array([0.001, -0.001, 0.0005, 0.01, -0.01, 0.02])
        neutral_gripper = composer.compose(wrist, 0.0)
        button_gripper = composer.compose(wrist, opening.value)
        np.testing.assert_allclose(button_gripper[composer.arm_slice], neutral_gripper[composer.arm_slice])
        self.assertNotEqual(
            float(button_gripper[composer.gripper_slice][0]),
            float(neutral_gripper[composer.gripper_slice][0]),
        )

    def test_jaco_held_button_moves_real_three_finger_gripper_then_release_is_zero(self):
        """A real Jaco environment receives multiple level-held commands."""

        env = self._env("Jaco")
        composer = OneDofGripperComposer(env)
        qpos_ids = np.asarray(
            env.robots[0]._ref_gripper_joint_pos_indexes[composer.gripper_key],
            dtype=np.intp,
        )
        initial = env.sim.data.qpos[qpos_ids].copy()
        controller = JacoContinuousGripperButtonController(
            self.config,
            open_command=composer.gripper_low,
            close_command=composer.gripper_high,
        )
        commands = []
        for _ in range(24):
            command = controller.update((True, False), valid=True, dt_s=1.0 / self.config.control_hz)
            commands.append(command.value)
            env.step(composer.compose(np.zeros(6), command.value))
        moved = env.sim.data.qpos[qpos_ids].copy()
        self.assertTrue(any(abs(value) > 0.0 for value in commands))
        self.assertGreater(np.linalg.norm(moved - initial), 1.0e-4)
        released = controller.update((False, False), valid=True, dt_s=1.0 / self.config.control_hz)
        self.assertEqual(released.value, 0.0)
        release_action = composer.compose(np.zeros(6), released.value)
        np.testing.assert_allclose(release_action[composer.arm_slice], 0.0)
        np.testing.assert_allclose(release_action[composer.gripper_slice], 0.0)


if __name__ == "__main__":
    unittest.main()
