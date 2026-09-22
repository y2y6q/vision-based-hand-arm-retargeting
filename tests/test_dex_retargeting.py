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
    return [Landmark(float(index) / 100.0, float(index) / 200.0, -float(index) / 300.0) for index in range(21)]


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
    def test_mediapipe_world_frame_is_wrist_relative_and_uses_documented_axes(self):
        points = hand()
        converted = mediapipe_world_to_mano(points, handedness="Right")
        self.assertEqual(converted.shape, (21, 3))
        np.testing.assert_allclose(converted[0], np.zeros(3))
        source_delta = np.array((points[4].x, points[4].y, points[4].z)) - np.array(
            (points[0].x, points[0].y, points[0].z)
        )
        np.testing.assert_allclose(converted[4], np.array((-source_delta[2], -source_delta[0], source_delta[1])))

    def test_handedness_and_shape_are_rejected_explicitly(self):
        with self.assertRaises(DexFrameRejected):
            mediapipe_world_to_mano(hand(), handedness="Left")
        with self.assertRaises(DexFrameRejected):
            mediapipe_world_to_mano(hand()[:20], handedness="Right")

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
            {"dex_hand": {"stale_timeout_s": 0.35, "expected_handedness": "Right", "input_is_mirrored": False}}
        )
        self.assertEqual(config.dex_hand.stale_timeout_s, 0.35)
        self.assertEqual(config.dex_hand.expected_handedness, "Right")

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
            )
            mano = np.arange(63, dtype=np.float64).reshape(21, 3) / 100.0
            result = adapter.retarget(mano)
            adapter.reset()

        expected_ref = mano[[4, 8, 12, 16]] - mano[[0, 0, 0, 0]]
        np.testing.assert_allclose(solver.retarget_arguments, [expected_ref])
        np.testing.assert_allclose(result.ref_value, expected_ref)
        np.testing.assert_allclose(result.raw_qpos, np.arange(16, dtype=np.float64))
        np.testing.assert_allclose(result.joint_targets_rad, np.arange(15, -1, -1, dtype=np.float64))
        self.assertEqual(result.retargeting_joint_names, solver_names)
        self.assertEqual(solver.reset_calls, 1)
        self.assertEqual(solver.filter.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
