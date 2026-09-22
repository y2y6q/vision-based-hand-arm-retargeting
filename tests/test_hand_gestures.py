"""Focused unit regressions for SpaceMouse button Allegro hand gestures."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.allegro_model import ALLEGRO_JOINT_NAMES, source_joint_specs  # noqa: E402
from wrist_teleop.hand_gestures import (  # noqa: E402
    CAMERA_SOURCE,
    FOUR_FINGER_PINCH,
    SAFE_OPEN_HOLD,
    THUMB_INDEX_PINCH,
    AllegroHandGestureController,
    HandGestureConfig,
)


class AllegroHandGestureControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = HandGestureConfig(interpolation_duration_s=0.20)
        self.controller = AllegroHandGestureController(self.config)
        self.camera = (self.controller.lower_limits + self.controller.upper_limits) * 0.5
        self.released = (False, False)
        self.left_pressed = (True, False)
        self.right_pressed = (False, True)

    def update(self, buttons, timestamp, *, valid=True, camera=None):
        return self.controller.update(
            buttons,
            self.camera if camera is None else camera,
            camera_valid=valid,
            timestamp_s=timestamp,
        )

    def test_targets_follow_actual_source_joint_order_and_limits(self):
        specs = source_joint_specs()
        self.assertEqual(self.controller.joint_names, ALLEGRO_JOINT_NAMES)
        self.assertEqual(self.controller.joint_names, tuple(spec.name for spec in specs))
        np.testing.assert_allclose(self.controller.lower_limits, [spec.lower for spec in specs])
        np.testing.assert_allclose(self.controller.upper_limits, [spec.upper for spec in specs])

        targets = self.controller.gesture_targets
        for target in targets.values():
            self.assertEqual(target.shape, (16,))
            self.assertTrue(np.all(np.isfinite(target)))
            self.assertTrue(np.all(target >= self.controller.lower_limits))
            self.assertTrue(np.all(target <= self.controller.upper_limits))

        open_target = targets[SAFE_OPEN_HOLD]
        thumb_index = targets[THUMB_INDEX_PINCH]
        four_finger = targets[FOUR_FINGER_PINCH]
        self.assertTrue(np.allclose(thumb_index[4:12], open_target[4:12]))
        self.assertTrue(np.any(thumb_index[:4] > open_target[:4]))
        self.assertTrue(np.any(thumb_index[12:16] > open_target[12:16]))
        self.assertTrue(np.any(four_finger[:4] > open_target[:4]))
        self.assertTrue(np.any(four_finger[4:8] > open_target[4:8]))
        self.assertTrue(np.any(four_finger[8:12] > open_target[8:12]))
        self.assertTrue(np.any(four_finger[12:16] > open_target[12:16]))

    def test_left_button_rising_edge_toggles_without_repeat_while_held(self):
        initial = self.update(self.released, 1.0)
        self.assertEqual(initial.source, CAMERA_SOURCE)
        self.assertTrue(initial.transition_active)
        initial = self.update(self.released, 1.21)
        self.assertFalse(initial.transition_active)
        np.testing.assert_allclose(initial.joint_targets, self.camera)

        clicked = self.update(self.left_pressed, 1.3)
        self.assertEqual(clicked.source, THUMB_INDEX_PINCH)
        self.assertEqual(clicked.active_override, THUMB_INDEX_PINCH)
        self.assertEqual(clicked.button_rising_edges, (0,))
        self.assertEqual(clicked.event, "activate_thumb_index_pinch")
        self.assertTrue(clicked.transition_active)
        # The output starts from the camera target, so the hand does not jump.
        np.testing.assert_allclose(clicked.joint_targets, self.camera)

        held = self.update(self.left_pressed, 1.4)
        self.assertEqual(held.active_override, THUMB_INDEX_PINCH)
        self.assertEqual(held.button_rising_edges, ())
        self.assertIsNone(held.event)
        self.assertTrue(held.transition_active)
        self.assertFalse(np.allclose(held.joint_targets, self.camera))

        settled = self.update(self.left_pressed, 1.51)
        self.assertFalse(settled.transition_active)
        np.testing.assert_allclose(
            settled.joint_targets, self.controller.gesture_targets[THUMB_INDEX_PINCH]
        )

        self.update(self.released, 1.52)
        cancelled = self.update(self.left_pressed, 1.53)
        self.assertIsNone(cancelled.active_override)
        self.assertEqual(cancelled.source, CAMERA_SOURCE)
        self.assertEqual(cancelled.event, "deactivate_thumb_index_pinch")

    def test_right_button_switches_and_second_click_returns_to_camera(self):
        self.update(self.released, 2.0)
        self.update(self.left_pressed, 2.1)
        self.update(self.released, 2.2)
        switched = self.update(self.right_pressed, 2.3)
        self.assertEqual(switched.source, FOUR_FINGER_PINCH)
        self.assertEqual(switched.active_override, FOUR_FINGER_PINCH)
        self.assertEqual(switched.event, "switch_thumb_index_pinch_to_four_finger_pinch")

        held = self.update(self.right_pressed, 2.4)
        self.assertEqual(held.active_override, FOUR_FINGER_PINCH)
        self.assertEqual(held.button_rising_edges, ())
        self.update(self.released, 2.5)
        cancelled = self.update(self.right_pressed, 2.6)
        self.assertIsNone(cancelled.active_override)
        self.assertEqual(cancelled.source, CAMERA_SOURCE)
        self.assertEqual(cancelled.event, "deactivate_four_finger_pinch")

    def test_invalid_camera_selects_safe_open_without_retaining_stale_target(self):
        self.update(self.released, 3.0)
        tracked = self.update(self.released, 3.21)
        self.assertEqual(tracked.source, CAMERA_SOURCE)
        invalid = self.update(self.released, 3.3, valid=False)
        self.assertEqual(invalid.source, SAFE_OPEN_HOLD)
        self.assertTrue(invalid.transition_active)
        # Complete the source-change transition and prove it did not keep the
        # formerly valid camera target indefinitely.
        safe = self.update(self.released, 3.51, valid=False)
        self.assertFalse(safe.transition_active)
        np.testing.assert_allclose(safe.joint_targets, self.controller.gesture_targets[SAFE_OPEN_HOLD])
        self.assertFalse(np.allclose(safe.joint_targets, tracked.joint_targets))

    def test_reset_clears_override_and_seeds_held_buttons(self):
        self.update(self.released, 4.0)
        self.update(self.left_pressed, 4.1)
        reset = self.controller.reset(buttons=self.left_pressed, timestamp_s=4.2)
        self.assertIsNone(reset.active_override)
        self.assertEqual(reset.source, SAFE_OPEN_HOLD)
        self.assertFalse(reset.transition_active)
        np.testing.assert_allclose(reset.joint_targets, self.controller.gesture_targets[SAFE_OPEN_HOLD])

        # A physical button held through reset is not a new click.
        still_held = self.update(self.left_pressed, 4.3, valid=False)
        self.assertIsNone(still_held.active_override)
        self.assertEqual(still_held.button_rising_edges, ())
        self.update(self.released, 4.4, valid=False)
        next_click = self.update(self.left_pressed, 4.5, valid=False)
        self.assertEqual(next_click.active_override, THUMB_INDEX_PINCH)

    def test_input_validation_rejects_invalid_config_and_button_report(self):
        with self.assertRaises(ValueError):
            HandGestureConfig(left_button_index=0, right_button_index=0)
        with self.assertRaises(ValueError):
            HandGestureConfig(thumb_index_pinch_curls=(0.0, 0.0, 0.0, 2.0))
        with self.assertRaises(ValueError):
            self.update((False,), 5.0)
        with self.assertRaises(ValueError):
            self.update((False, 1), 5.0)


if __name__ == "__main__":
    unittest.main()
