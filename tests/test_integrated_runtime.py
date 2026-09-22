"""Headless reset, keyboard command, and shutdown regressions."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop import HandInputAdapter, SixDofMapper, WristConfig  # noqa: E402
from wrist_teleop.allegro_model import make_allegro_panda_env  # noqa: E402
from wrist_teleop.integrated_runtime import (  # noqa: E402
    close_integrated_resources,
    reset_integrated_environment,
)
from wrist_teleop.runtime_controls import KeyboardCommandRouter, install_viewer_key_callback  # noqa: E402


class FakeDevice:
    def __init__(self):
        self.reset_calls = 0
        self.closed = False

    def reset_input_state(self):
        self.reset_calls += 1

    def close(self):
        self.closed = True


class FakeCamera:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class RuntimeControlTests(unittest.TestCase):
    def test_r_is_one_shot_and_q_sets_unified_stop_event(self):
        router = KeyboardCommandRouter(repeat_release_s=0.25)
        self.assertEqual(router.feed("r", timestamp_s=1.0), "reset")
        self.assertTrue(router.take_reset())
        self.assertIsNone(router.feed(ord("R"), timestamp_s=1.05))
        self.assertFalse(router.take_reset())
        self.assertEqual(router.feed("r", timestamp_s=1.40), "reset")
        self.assertTrue(router.take_reset())
        self.assertEqual(router.feed("q", source="camera", timestamp_s=2.0), "quit")
        self.assertTrue(router.stop_event.is_set())
        self.assertEqual(router.quit_reason, "camera:Q")
        self.assertIsNone(router.feed("q", timestamp_s=2.01))

    def test_non_mjviewer_callback_path_routes_camera_or_viewer_key(self):
        class Viewer:
            def add_keypress_callback(self, callback):
                self.callback = callback

        class Env:
            viewer = Viewer()

        router = KeyboardCommandRouter()
        env = Env()
        self.assertTrue(install_viewer_key_callback(env, lambda key: router.feed(key, source="viewer")))
        env.viewer.callback(ord("Q"))
        self.assertTrue(router.stop_event.is_set())
        self.assertEqual(router.quit_reason, "viewer:Q")

    def test_local_mjviewer_wrapper_passes_r_to_native_key_callback(self):
        class MjviewerRenderer:
            def __init__(self):
                self.viewer = None
                self.camera_id = None
                self.camera_config = {"lookat": [0, 0, 1], "distance": 2, "azimuth": 180, "elevation": -20}

            def close(self):
                pass

        class NativeHandle:
            def __init__(self):
                self.opt = type("Opt", (), {"geomgroup": [1]})()
                self.cam = type("Cam", (), {})()
                self.synced = False

            def sync(self):
                self.synced = True

            def close(self):
                pass

        class Sim:
            model = type("Model", (), {"_model": object()})()
            data = type("Data", (), {"_data": object()})()

        class Env:
            def __init__(self):
                self.viewer = MjviewerRenderer()
                self.sim = Sim()

        router = KeyboardCommandRouter()
        native = NativeHandle()
        env = Env()
        with patch("mujoco.viewer.launch_passive", return_value=native) as launch:
            self.assertTrue(install_viewer_key_callback(env, lambda key: router.feed(key, source="mujoco")))
            env.viewer.update()
        self.assertIs(launch.call_args.kwargs["key_callback"], env.viewer.keypress_callback)
        env.viewer.keypress_callback(ord("R"))
        self.assertTrue(router.take_reset())
        self.assertTrue(native.synced)

    def test_headless_reset_clears_filters_and_override_then_shutdowns_with_zero_arm(self):
        config = WristConfig(smoothing_alpha=0.35)
        env = make_allegro_panda_env(config, has_renderer=False)
        mapper = SixDofMapper(config)
        device = FakeDevice()
        hand_input = HandInputAdapter()
        camera = FakeCamera()
        try:
            composer, gestures, initial_hand, initial_action = reset_integrated_environment(
                env,
                mapper=mapper,
                device=device,
                hand_input=hand_input,
                gestures=None,
                gesture_config=config.hand_gestures,
            )
            self.assertEqual(device.reset_calls, 1)
            self.assertIsNone(initial_hand.active_override)
            np.testing.assert_allclose(initial_action[composer.arm_slice], np.zeros(6))

            mapper.map([350, 0, 0, 0, 0, 0], valid=True)
            gestures.update((False, False), initial_hand.joint_targets, camera_valid=False, timestamp_s=1.0)
            active = gestures.update((True, False), initial_hand.joint_targets, camera_valid=False, timestamp_s=1.3)
            self.assertEqual(active.active_override, "thumb_index_pinch")

            composer, gestures, reset_hand, reset_action = reset_integrated_environment(
                env,
                mapper=mapper,
                device=device,
                hand_input=hand_input,
                gestures=gestures,
                buttons=(True, False),
            )
            self.assertEqual(device.reset_calls, 2)
            self.assertIsNone(reset_hand.active_override)
            self.assertEqual(reset_hand.source, "safe_open/hold")
            np.testing.assert_allclose(reset_action[composer.arm_slice], np.zeros(6))
            np.testing.assert_allclose(mapper.map(np.zeros(6), valid=True), np.zeros(6))
            self.assertEqual(hand_input.latest().tracking_state, "waiting_for_hand")

            last_action = close_integrated_resources(
                env=env,
                composer=composer,
                mapper=mapper,
                device=device,
                camera=camera,
            )
            env = None
            self.assertTrue(device.closed)
            self.assertTrue(camera.closed)
            np.testing.assert_allclose(last_action[composer.arm_slice], np.zeros(6))
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    unittest.main()
