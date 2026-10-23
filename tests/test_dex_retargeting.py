"""Dependency-safe regression tests for the official dex integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.dex_retargeting import (  # noqa: E402
    DexFrameRejected,
    DexRetargetingAdapter,
    DexRetargetingResult,
    DexRetargetingUnavailable,
    mediapipe_world_to_mano,
)
from wrist_teleop.hand_input import DexHandInputAdapter  # noqa: E402
from wrist_teleop.config import WristConfig  # noqa: E402


@dataclass
class Landmark:
    x: float
    y: float
    z: float


def hand() -> list[Landmark]:
    """A non-degenerate open right hand in MediaPipe world coordinates."""

    values = np.array(
        (
            (0.000, 0.000, 0.000),  # wrist
            (0.020, -0.004, 0.005), (0.030, 0.006, 0.007), (0.040, 0.016, 0.006), (0.050, 0.024, 0.004),
            (0.030, 0.000, 0.000), (0.034, 0.026, 0.003), (0.036, 0.050, 0.005), (0.038, 0.070, 0.006),
            (0.000, 0.050, 0.000), (0.000, 0.080, 0.002), (0.000, 0.105, 0.003), (0.000, 0.125, 0.004),
            (-0.026, 0.040, 0.000), (-0.030, 0.066, 0.002), (-0.032, 0.088, 0.003), (-0.034, 0.106, 0.004),
            (-0.044, 0.012, 0.000), (-0.052, 0.032, 0.002), (-0.057, 0.049, 0.003), (-0.060, 0.064, 0.004),
        ),
        dtype=np.float64,
    )
    return [Landmark(*point) for point in values]


class FakeRetargeter:
    def __init__(self):
        self.reset_calls = 0

    def retarget(self, _mano):
        values = np.linspace(-0.5, 2.0, 16)
        return DexRetargetingResult(
            joint_targets_rad=values,
            raw_qpos=values + 0.1,
            ref_value=np.ones((4, 3)),
            human_indices=np.array(((0, 0, 0, 0), (4, 8, 12, 16))),
            retargeting_joint_names=tuple(f"joint_{index}.0" for index in range(16)),
        )

    def reset(self):
        self.reset_calls += 1


class DexRetargetingTests(unittest.TestCase):
    def test_mediapipe_world_frame_matches_official_wrist_frame_convention(self):
        points = hand()
        converted = mediapipe_world_to_mano(points, handedness="Right")
        self.assertEqual(converted.shape, (21, 3))
        np.testing.assert_allclose(converted[0], np.zeros(3))
        # In this synthetic palm, the middle MCP lies on the official MANO
        # +Z direction, while index MCP lies on +Y after palm normalization.
        np.testing.assert_allclose(converted[9], np.array((0.0, 0.0, 0.05)), atol=1e-12)
        np.testing.assert_allclose(converted[5], np.array((0.0, 0.03, 0.0)), atol=1e-12)

    def test_palm_local_mano_frame_is_invariant_to_webcam_orientation(self):
        points = hand()
        source = np.array([(point.x, point.y, point.z) for point in points])
        # Simulate the same hand seen after a 90-degree camera-frame yaw.
        camera_rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        rotated = [Landmark(*point) for point in source @ camera_rotation]
        np.testing.assert_allclose(
            mediapipe_world_to_mano(rotated, handedness="Right"),
            mediapipe_world_to_mano(points, handedness="Right"),
            atol=1e-12,
        )

    def test_handedness_and_shape_are_rejected_explicitly(self):
        with self.assertRaises(DexFrameRejected):
            mediapipe_world_to_mano(hand(), handedness="Left")
        with self.assertRaises(DexFrameRejected):
            mediapipe_world_to_mano(hand()[:20], handedness="Right")

    def test_degenerate_palm_is_rejected_before_it_can_move_the_robot_hand(self):
        collinear = [Landmark(float(index), 0.0, 0.0) for index in range(21)]
        with self.assertRaisesRegex(DexFrameRejected, "degenerate"):
            mediapipe_world_to_mano(collinear, handedness="Right")

    def test_adapter_clips_named_dex_targets_and_resets_solver_history(self):
        lower = np.array([-0.2] * 16)
        upper = np.array([0.8] * 16)
        solver = FakeRetargeter()
        adapter = DexHandInputAdapter(
            solver,
            lower_limits=lower,
            upper_limits=upper,
            joint_names=tuple(f"joint_{index}.0" for index in range(16)),
            stale_timeout_s=0.2,
        )
        command = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.0)
        self.assertTrue(command.valid)
        self.assertEqual(command.details["algorithm"], "dex-retargeting")
        self.assertIn("raw_dex_qpos", command.details)
        self.assertTrue(np.all(command.joint_targets >= lower))
        self.assertTrue(np.all(command.joint_targets <= upper))
        stale = adapter.latest(timestamp_s=10.3)
        self.assertFalse(stale.valid)
        reset = adapter.reset(timestamp_s=11.0)
        self.assertFalse(reset.valid)
        self.assertEqual(solver.reset_calls, 1)
        self.assertEqual(reset.details["history_reset"], True)

    def test_relative_camera_neutral_maps_official_pose_changes_around_safe_open(self):
        """The first valid right-hand pose is neutral; later dex motion remains visible."""

        class ChangingSolver(FakeRetargeter):
            def __init__(self):
                super().__init__()
                self.offset = 0.0

            def retarget(self, _mano):
                values = np.linspace(-0.5, 0.5, 16) + self.offset
                return DexRetargetingResult(
                    joint_targets_rad=values,
                    raw_qpos=values,
                    ref_value=np.ones((4, 3)),
                    human_indices=np.array(((0, 0, 0, 0), (4, 8, 12, 16))),
                    retargeting_joint_names=tuple(f"joint_{index}.0" for index in range(16)),
                )

        solver = ChangingSolver()
        lower = np.array([-2.0] * 16)
        upper = np.array([2.0] * 16)
        adapter = DexHandInputAdapter(
            solver, lower_limits=lower, upper_limits=upper,
            joint_names=tuple(f"joint_{index}.0" for index in range(16)),
            relative_motion=True, relative_motion_gain=1.0,
        )
        neutral = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.0)
        np.testing.assert_allclose(neutral.joint_targets, np.zeros(16))
        self.assertEqual(neutral.details["relative_reference_event"], "captured_first_valid_frame")
        solver.offset = 0.25
        moved = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.1)
        np.testing.assert_allclose(moved.joint_targets, np.full(16, 0.25))
        self.assertTrue(adapter.recenter_relative_motion())
        solver.offset = 0.50
        moved_after_recenter = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.2)
        np.testing.assert_allclose(moved_after_recenter.joint_targets, np.full(16, 0.25))

    def test_relative_camera_neutral_can_be_disabled_for_absolute_official_targets(self):
        adapter = DexHandInputAdapter(
            FakeRetargeter(), lower_limits=np.array([-2.0] * 16), upper_limits=np.array([2.0] * 16),
            joint_names=tuple(f"joint_{index}.0" for index in range(16)), relative_motion=False,
        )
        command = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.0)
        self.assertFalse(command.details["relative_motion"])
        self.assertFalse(adapter.recenter_relative_motion())
        self.assertEqual(adapter.calibration_state, "absolute official dex vector target")

    def test_detector_loss_holds_once_then_expires_to_safe_open_policy(self):
        adapter = DexHandInputAdapter(
            FakeRetargeter(),
            lower_limits=np.array([-1.0] * 16),
            upper_limits=np.array([1.0] * 16),
            joint_names=tuple(f"joint_{index}.0" for index in range(16)),
            stale_timeout_s=0.2,
        )
        accepted = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.0)
        held = adapter.mark_lost(timestamp_s=10.1, error="no_hand")
        self.assertTrue(held.valid)
        np.testing.assert_allclose(held.joint_targets, accepted.joint_targets)
        self.assertEqual(held.details["loss_policy"], "hold_last_valid_then_safe_open")
        expired = adapter.latest(timestamp_s=10.21)
        self.assertFalse(expired.valid)
        self.assertEqual(expired.tracking_state, "stale_safe_open")

    def test_current_environment_reports_missing_dex_without_legacy_fallback(self):
        with self.assertRaises(DexRetargetingUnavailable) as captured:
            DexRetargetingAdapter(
                allegro_joint_names=tuple(f"joint_{index}.0" for index in range(16)),
                lower_limits=[-1.0] * 16,
                upper_limits=[1.0] * 16,
                runtime="native",
        )
        self.assertIn("not installed", str(captured.exception))
        self.assertIn("pin>=2.7", str(captured.exception))

    def test_dex_camera_loss_policy_comes_from_the_shared_wrist_config(self):
        config = WristConfig.from_dict(
            {"dex_hand": {
                "stale_timeout_s": 0.35, "expected_handedness": "Right", "input_is_mirrored": False,
                "relative_motion": True, "relative_motion_gain": 1.25,
            }}
        )
        self.assertEqual(config.dex_hand.stale_timeout_s, 0.35)
        self.assertEqual(config.dex_hand.expected_handedness, "Right")
        self.assertTrue(config.dex_hand.relative_motion)
        self.assertEqual(config.dex_hand.relative_motion_gain, 1.25)

    def test_dex_frame_path_does_not_invoke_the_legacy_curl_estimator(self):
        """A valid dex frame must reach only the solver / named-joint path."""

        adapter = DexHandInputAdapter(
            FakeRetargeter(),
            lower_limits=np.array([-1.0] * 16),
            upper_limits=np.array([1.0] * 16),
            joint_names=tuple(f"joint_{index}.0" for index in range(16)),
        )
        with patch(
            "wrist_teleop.hand_input.estimate_all_finger_curls",
            side_effect=AssertionError("legacy curl must not be called by dex"),
        ):
            command = adapter.update_world_landmarks(hand(), "Right", timestamp_s=10.0)
        self.assertTrue(command.valid)
        self.assertEqual(command.details["algorithm"], "dex-retargeting")

    def test_official_boundary_builds_vector_solver_and_maps_every_named_joint(self):
        """Exercise the package boundary with an official-API-shaped fake.

        The current Windows interpreter cannot import Pinocchio, so this
        verifies the project's code around the actual 0.4.6 public API while
        keeping the hardware / native-package claim separate.
        """

        local_names = tuple(f"joint_{index}.0" for index in range(16))
        solver_names = tuple(reversed(local_names))

        class FakeFilter:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

        class FakeSolver:
            def __init__(self):
                self.optimizer = types.SimpleNamespace(
                    target_link_human_indices=np.array(((0, 0, 0, 0), (4, 8, 12, 16)))
                )
                self.joint_names = solver_names
                self.filter = FakeFilter()
                self.retarget_arguments = []
                self.reset_calls = 0

            def retarget(self, reference):
                self.retarget_arguments.append(np.asarray(reference).copy())
                return np.arange(16, dtype=np.float64)

            def reset(self):
                self.reset_calls += 1

        solver = FakeSolver()

        class FakeLoadedConfig:
            def build(self):
                return solver

        class FakeConfig:
            urdf_root = None

            @classmethod
            def set_default_urdf_dir(cls, value):
                cls.urdf_root = value

            @classmethod
            def load_from_file(cls, _path):
                return FakeLoadedConfig()

        package = types.ModuleType("dex_retargeting")
        module = types.ModuleType("dex_retargeting.retargeting_config")
        module.RetargetingConfig = FakeConfig
        with patch.dict(sys.modules, {
            "dex_retargeting": package,
            "dex_retargeting.retargeting_config": module,
        }), patch("wrist_teleop.dex_retargeting.importlib.metadata.version", return_value="0.4.6"):
            adapter = DexRetargetingAdapter(
                allegro_joint_names=local_names,
                lower_limits=[-100.0] * 16,
                upper_limits=[100.0] * 16,
                runtime="native",
                reference_landmark_scale=2.0 / 3.0,
            )
            mano = np.arange(63, dtype=np.float64).reshape(21, 3) / 100.0
            result = adapter.retarget(mano)
            adapter.reset()

        expected_ref = (mano[[4, 8, 12, 16]] - mano[[0, 0, 0, 0]]) * (2.0 / 3.0)
        np.testing.assert_allclose(solver.retarget_arguments, [expected_ref])
        np.testing.assert_allclose(result.ref_value, expected_ref)
        np.testing.assert_allclose(result.raw_qpos, np.arange(16, dtype=np.float64))
        np.testing.assert_allclose(result.joint_targets_rad, np.arange(15, -1, -1, dtype=np.float64))
        self.assertEqual(result.retargeting_joint_names, solver_names)
        self.assertEqual(adapter.metadata()["reference_landmark_scale"], 2.0 / 3.0)
        self.assertEqual(solver.reset_calls, 1)
        self.assertEqual(solver.filter.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
