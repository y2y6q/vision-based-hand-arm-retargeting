"""Hardware-free safety and physical-unit checks for wrist-only mapping."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_teleop import SixDofMapper, WristConfig
from wrist_teleop.mapper import camera_frame_to_world_rotation


def config(**overrides) -> WristConfig:
    settings = dict(axis_order=(0, 1, 2, 3, 4, 5), axis_inversion=(1,) * 6,
                    translation_deadzone=0.0, rotation_deadzone=0.0,
                    translation_release_deadzone=0.0, rotation_release_deadzone=0.0,
                    translation_scale_m=0.01, rotation_scale_rad=0.1,
                    max_translation_delta_m=1.0, max_rotation_delta_rad=1.0,
                    smoothing_alpha=None, control_frame="world")
    settings.update(overrides)
    return WristConfig(**settings)


class MapperTests(unittest.TestCase):
    def test_deadzone_and_boundary_are_exact_zero(self):
        mapper = SixDofMapper(config(raw_ranges=(100,) * 6,
                                    translation_deadzone=0.1, rotation_deadzone=0.2))
        for raw in ([0] * 6, [10, -10, 9, 20, -20, 19], [1, 2, -1, 2, 3, -2]):
            np.testing.assert_array_equal(mapper.map(raw), np.zeros(6))

    def test_deadzone_rescales_continuously(self):
        mapper = SixDofMapper(config(raw_ranges=(100,) * 6,
                                    translation_deadzone=0.1, rotation_deadzone=0.2))
        np.testing.assert_allclose(mapper.map([55, -55, 0, 60, -60, 0]),
                                   [0.005, -0.005, 0, 0.05, -0.05, 0], atol=1e-15)

    def test_translation_and_rotation_have_independent_scales(self):
        mapper = SixDofMapper(config(translation_scale_m=0.007, rotation_scale_rad=0.03))
        np.testing.assert_allclose(mapper.map([350, 0, 0, 350, 0, 0]),
                                   [0.007, 0, 0, 0.03, 0, 0])

    def test_raw_ranges_apply_before_axis_reorder(self):
        mapper = SixDofMapper(config(raw_ranges=(100, 200, 300, 400, 500, 600),
                                    axis_order=(1, 0, 2, 5, 3, 4)))
        np.testing.assert_allclose(mapper.map([100, 100, -300, 200, 500, -300]),
                                   [0.005, 0.01, -0.01, -0.05, 0.05, 0.1])

    def test_axis_permutation_and_sign_are_explicit(self):
        mapper = SixDofMapper(config(axis_order=(1, 0, 2, 4, 5, 3),
                                    axis_inversion=(1, -1, 1, -1, 1, -1)))
        np.testing.assert_allclose(mapper.map([35, 70, 105, 140, 175, 210]),
                                   [0.002, -0.001, 0.003, -0.05, 0.06, -0.04])

    def test_world_rotation_applies_to_both_triads(self):
        mapper = SixDofMapper(config(device_to_world=((0, -1, 0), (1, 0, 0), (0, 0, 1))))
        np.testing.assert_allclose(mapper.map([350, 0, 0, 350, 0, 0]),
                                   [0, 0.01, 0, 0, 0.1, 0])

    def test_camera_frame_maps_each_of_six_unit_axes_from_real_extrinsics(self):
        # MuJoCo's columns are camera right, up, and back. This upright camera
        # looks along world -X, so expected controls are right=+Y,
        # depth=-X, vertical=+Z for both translation and rotation.
        world_from_camera = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        expected_basis = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        np.testing.assert_allclose(camera_frame_to_world_rotation(world_from_camera), expected_basis)
        mapper = SixDofMapper(config(control_frame="camera"))
        record = mapper.set_camera_extrinsics(world_from_camera, camera_name="frontview")
        self.assertEqual(record["source"], "camera_extrinsics")
        self.assertEqual(record["basis_axes"], ["screen_right", "view_depth", "world_up"])
        for axis in range(6):
            with self.subTest(axis=axis):
                mapper.reset()
                raw = np.zeros(6)
                raw[axis] = 350.0
                expected = np.zeros(6)
                expected[:3] = expected_basis[:, axis] * 0.01 if axis < 3 else 0.0
                expected[3:] = expected_basis[:, axis - 3] * 0.1 if axis >= 3 else 0.0
                np.testing.assert_allclose(mapper.map(raw), expected, atol=1e-15)

    def test_camera_frame_can_bind_live_cam_xmat(self):
        world_from_camera = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

        class Model:
            @staticmethod
            def camera_name2id(name):
                return 0 if name == "frontview" else -1

        class Data:
            cam_xmat = world_from_camera.reshape(1, 9)

        class Sim:
            model = Model()
            data = Data()

        mapper = SixDofMapper(config(control_frame="camera"))
        record = mapper.set_camera_from_sim(Sim())
        self.assertEqual(record["source"], "live_mujoco_cam_xmat")
        self.assertEqual(record["primary_camera_name"], "frontview")
        np.testing.assert_allclose(mapper.control_rotation, camera_frame_to_world_rotation(world_from_camera))

    def test_translation_and_rotation_clamps_preserve_direction(self):
        mapper = SixDofMapper(config(max_translation_delta_m=0.004,
                                    max_rotation_delta_rad=0.04))
        command = mapper.map([350, -350, 350, -350, 350, -350])
        self.assertAlmostEqual(float(np.linalg.norm(command[:3])), 0.004)
        self.assertAlmostEqual(float(np.linalg.norm(command[3:])), 0.04)
        np.testing.assert_allclose(command[:3] / command[0], [1, -1, 1])
        np.testing.assert_allclose(command[3:] / command[3], [1, -1, 1])

    def test_extreme_finite_input_is_clamped(self):
        mapper = SixDofMapper(config(raw_ranges=(1e-300,) * 6,
                                    max_translation_delta_m=0.004,
                                    max_rotation_delta_rad=0.04))
        with np.errstate(all="raise"):
            command = mapper.map([1e300, -1e300, 1e300, -1e300, 1e300, -1e300])
        self.assertTrue(np.all(np.isfinite(command)))
        self.assertLessEqual(np.linalg.norm(command[:3]), 0.004 + 1e-15)
        self.assertLessEqual(np.linalg.norm(command[3:]), 0.04 + 1e-15)

    def test_ema_and_explicit_reset(self):
        mapper = SixDofMapper(config(smoothing_alpha=0.25))
        raw = [350, 0, 0, 350, 0, 0]
        first = mapper.map(raw)
        np.testing.assert_allclose(first, [0.0025, 0, 0, 0.025, 0, 0])
        np.testing.assert_allclose(mapper.map(raw), first * 1.75)
        mapper.reset()
        np.testing.assert_allclose(mapper.map(raw), first)

    def test_release_resets_each_axis_without_affecting_active_axes(self):
        mapper = SixDofMapper(config(smoothing_alpha=0.25))
        mapper.map([350] * 6)
        command = mapper.map([0, 350, 0, 350, 0, 350])
        np.testing.assert_array_equal(command[[0, 2, 4]], np.zeros(3))
        np.testing.assert_allclose(command[[1, 3, 5]], [0.004375, 0.04375, 0.04375])
        command = mapper.map([350, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(command[0], 0.0025)
        np.testing.assert_array_equal(command[1:], np.zeros(5))

    def test_neutral_noise_after_motion_has_no_filter_tail_or_drift(self):
        mapper = SixDofMapper(WristConfig())
        for _ in range(20):
            mapper.map([350] * 6)
        accumulated = np.zeros(6)
        for i in range(1000):
            command = mapper.map([(-1) ** i * 20] * 6)
            np.testing.assert_array_equal(command, np.zeros(6))
            accumulated += command
        np.testing.assert_array_equal(accumulated, np.zeros(6))

    def test_hysteresis_and_consecutive_neutral_lock_clear_ema_exactly(self):
        mapper = SixDofMapper(config(
            raw_ranges=(100,) * 6,
            translation_deadzone=0.10,
            rotation_deadzone=0.10,
            translation_release_deadzone=0.05,
            rotation_release_deadzone=0.05,
            neutral_lock_frames=3,
            smoothing_alpha=0.25,
        ))
        self.assertGreater(mapper.map([50, 0, 0, 0, 0, 0])[0], 0.0)
        # Below the enter threshold but above release leaves the latch on yet
        # emits no residual filtered action.
        np.testing.assert_array_equal(mapper.map([8, 0, 0, 0, 0, 0]), np.zeros(6))
        self.assertTrue(mapper.diagnostics()["axis_active"][0])
        for _ in range(3):
            np.testing.assert_array_equal(mapper.map([4, 0, 0, 0, 0, 0]), np.zeros(6))
        status = mapper.diagnostics()
        self.assertTrue(status["neutral_locked"])
        self.assertEqual(status["ema_axes"], [0.0] * 6)
        self.assertEqual(status["axis_active"], [False] * 6)

    def test_invalid_samples_zero_and_reset_history(self):
        mapper = SixDofMapper(config(smoothing_alpha=0.25))
        raw = [350, 0, 0, 0, 0, 0]
        expected_first = mapper.map(raw)
        invalid_samples = [None, [], [1] * 5, [1] * 7, [[1] * 6],
                           [1, 2, 3, 4, 5, np.nan], [np.inf] * 6, [True] * 6,
                           ["1"] * 6, [1 + 2j] * 6, object(), [[1], 2, 3, 4, 5, 6]]
        for invalid in invalid_samples:
            with self.subTest(raw=invalid):
                mapper.map(raw)
                np.testing.assert_array_equal(mapper.map(invalid), np.zeros(6))
                np.testing.assert_allclose(mapper.map(raw), expected_first)
        for validity in (False, None, 1, "valid"):
            with self.subTest(valid=validity):
                mapper.map(raw)
                np.testing.assert_array_equal(mapper.map(raw, valid=validity), np.zeros(6))
                np.testing.assert_allclose(mapper.map(raw), expected_first)

    def test_translation_and_rotation_cannot_cross_control_chains(self):
        mapper = SixDofMapper(WristConfig())
        for i in range(6):
            mapper.reset()
            raw = np.zeros(6)
            raw[i] = 350
            command = mapper.map(raw)
            self.assertGreater(np.linalg.norm(command), 0)
            np.testing.assert_array_equal(command[3:] if i < 3 else command[:3], np.zeros(3))

    def test_return_value_does_not_expose_filter_state(self):
        mapper = SixDofMapper(config(smoothing_alpha=0.25))
        command = mapper.map([350, 0, 0, 0, 0, 0])
        command[:] = 999
        self.assertAlmostEqual(mapper.map([350, 0, 0, 0, 0, 0])[0], 0.004375)


class ConfigTests(unittest.TestCase):
    def test_repository_json_matches_defaults(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "spacemouse_wrist.json"
        self.assertEqual(WristConfig.load(path), WristConfig())
        self.assertEqual(WristConfig.load(), WristConfig())

    def test_json_roundtrip_and_partial_overrides(self):
        original = replace(WristConfig(), translation_deadzone=0.2, smoothing_alpha=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
            self.assertEqual(WristConfig.load(path), original)
        self.assertEqual(WristConfig.from_dict({"control_hz": 30}).control_hz, 30)

    def test_settings_are_immutable(self):
        inversion = [1] * 6
        settings = WristConfig(axis_inversion=inversion)
        inversion[0] = -1
        self.assertEqual(settings.axis_inversion[0], 1)
        with self.assertRaises(FrozenInstanceError):
            settings.control_hz = 60

    def test_invalid_numeric_settings(self):
        positive = ("translation_scale_m", "rotation_scale_rad", "max_translation_delta_m",
                    "max_rotation_delta_rad", "control_hz", "stale_timeout_s",
                    "discovery_interval_s", "neutral_startup_s")
        for field in positive:
            for value in (0, -1, float("nan"), float("inf"), True, "1", None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    WristConfig.from_dict({field: value})
        for field in ("translation_deadzone", "rotation_deadzone",
                      "translation_release_deadzone", "rotation_release_deadzone"):
            for value in (-0.1, 1.0, float("nan"), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    WristConfig.from_dict({field: value})
        with self.assertRaises(ValueError):
            WristConfig(translation_deadzone=0.05, translation_release_deadzone=0.06)
        for field in ("zero_bias_calibration_frames", "neutral_lock_frames"):
            for value in (0, -1, 1.5, True, "3", 121):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    WristConfig.from_dict({field: value})
        for value in (0, -1, 1.01, float("inf"), True):
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                WristConfig(smoothing_alpha=value)

    def test_invalid_axes_and_cross_triad_permutations(self):
        cases = [{"axis_order": [0, 1, 3, 2, 4, 5]}, {"axis_order": [0, 0, 2, 3, 4, 5]},
                 {"axis_order": [0., 1., 2., 3., 4., 5.]}, {"axis_order": [False, 1, 2, 3, 4, 5]},
                 {"axis_order": [0, 1, 2]}, {"axis_inversion": [0] * 6},
                 {"axis_inversion": [True] * 6}, {"raw_ranges": [0] * 6},
                 {"raw_ranges": [1, 1, 1, 1, 1, float("nan")]}, {"raw_ranges": "350350"}]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                WristConfig.from_dict(values)

    def test_world_transform_must_be_a_proper_rotation(self):
        for matrix in (((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
                       ((2, 0, 0), (0, 1, 0), (0, 0, 1)), np.eye(2),
                       ((float("nan"), 0, 0), (0, 1, 0), (0, 0, 1))):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                WristConfig(device_to_world=matrix)
        with self.assertRaises(ValueError):
            WristConfig(coordinate_frame="tool")
        with self.assertRaises(ValueError):
            WristConfig(control_frame="tool")

    def test_invalid_lifecycle_and_unknown_settings(self):
        for values in ({"vendor_id": -1}, {"product_id": 65536},
                       {"hand_gestures": {"left_button_index": 32}},
                       {"hand_gestures": {"left_button_index": True}},
                       {"output_dir": ""}, {"output_dir": None},
                       {"output_dir": "bad\x00path"}, {"translation_sensitivity": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                WristConfig.from_dict(values)
        with self.assertRaises(ValueError):
            WristConfig.from_dict([])


if __name__ == "__main__":
    unittest.main()
