"""Guided SpaceMouse direction calibration solver (no hardware)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrist_teleop.axis_calibration import AXIS_GESTURES, measure_gesture, solve_axis_mapping  # noqa: E402
from wrist_teleop.config import WristConfig  # noqa: E402

# Raw HID response of each gesture under the 3Dconnexion convention
# (+X right, +Y toward the operator, +Z down, right-handed rotations).
HID_CONVENTION = {
    "push_away": (1, -1),
    "push_right": (0, +1),
    "lift_up": (2, -1),
    "tilt_away": (3, -1),
    "tilt_right": (4, -1),
    "twist_clockwise": (5, +1),
}


def _samples(axis: int, sign: int, leak: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(axis)
    values = rng.normal(0.0, 5.0, size=(40, 6))
    values[:, axis] += sign * 250.0
    values[:, (axis + 1) % 3 + (3 if axis >= 3 else 0)] += leak * 250.0
    return values


class AxisCalibrationTests(unittest.TestCase):
    def test_hid_convention_solves_to_the_shipped_defaults(self):
        measurements = [measure_gesture(g, _samples(*HID_CONVENTION[g.key])) for g in AXIS_GESTURES]
        order, inversion = solve_axis_mapping(measurements)
        defaults = WristConfig()
        self.assertEqual(order, defaults.axis_order)
        self.assertEqual(inversion, defaults.axis_inversion)

    def test_rotated_device_is_recovered(self):
        # Device turned 90 degrees: "away" now arrives on raw X, "right" on raw Y.
        swapped = dict(HID_CONVENTION, push_away=(0, +1), push_right=(1, +1))
        order, inversion = solve_axis_mapping(
            [measure_gesture(g, _samples(*swapped[g.key])) for g in AXIS_GESTURES]
        )
        self.assertEqual(order[:3], (1, 0, 2))
        self.assertEqual(inversion[:2], (1, 1))

    def test_mixed_or_duplicate_gestures_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_axis_mapping([measure_gesture(AXIS_GESTURES[0], _samples(1, -1, leak=0.9))])
        same = [measure_gesture(g, _samples(0, +1) if g.control_index < 3 else _samples(*HID_CONVENTION[g.key]))
                for g in AXIS_GESTURES]
        with self.assertRaises(ValueError):
            solve_axis_mapping(same)
        with self.assertRaises(ValueError):
            measure_gesture(AXIS_GESTURES[0], np.zeros((10, 6)))


if __name__ == "__main__":
    unittest.main()
