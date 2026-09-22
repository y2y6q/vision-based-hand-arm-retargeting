"""Hardware-free HID packet and fail-closed tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_teleop import SpaceMouseHid, WristConfig


def packet(report_id, *values):
    return [report_id, *values]


class FakeDevice:
    def __init__(self, reports=(), read_error=None):
        self.reports = list(reports)
        self.read_error = read_error
        self.opened_path = None
        self.closed = False
        self.nonblocking = None

    def open_path(self, path):
        self.opened_path = path

    def set_nonblocking(self, value):
        self.nonblocking = value

    def read(self, _size):
        if self.read_error is not None:
            raise self.read_error
        return self.reports.pop(0) if self.reports else []

    def close(self):
        self.closed = True


class FakeHid:
    def __init__(self, device):
        self.device_instance = device

    @staticmethod
    def enumerate():
        return [{
            "path": r"\\?\HID#VID_256F&PID_C62E#fake",
            "vendor_id": 0x256F,
            "product_id": 0xC62E,
            "manufacturer_string": "3Dconnexion",
            "product_string": "SpaceMouse Wireless",
        }]

    def device(self):
        return self.device_instance


class SpaceMouseHidTests(unittest.TestCase):
    def make_reader(self, reports=(), read_error=None):
        fake_device = FakeDevice(reports, read_error)
        reader = SpaceMouseHid(WristConfig(neutral_startup_s=1e-9), hid_module=FakeHid(fake_device))
        reader.open()
        return reader, fake_device

    def test_discovery_and_windows_string_path_are_supported(self):
        reader, fake = self.make_reader()
        self.assertEqual(reader.device_info["product_id"], 0xC62E)
        self.assertIsInstance(fake.opened_path, bytes)
        self.assertEqual(fake.nonblocking, 1)
        reader.close()
        self.assertTrue(fake.closed)

    def test_full_six_axis_report_keeps_raw_report_order(self):
        reader, _ = self.make_reader([
            packet(1, 1, 0, 2, 0, 253, 255, 4, 0, 251, 255, 6, 0),
        ])
        sample = reader.poll()
        np.testing.assert_array_equal(sample.raw_axes, [1, 2, -3, 4, -5, 6])
        self.assertTrue(sample.valid)

    def test_split_translation_rotation_and_button_packets(self):
        reader, _ = self.make_reader([
            packet(1, 10, 0, 20, 0, 30, 0),
            packet(2, 246, 255, 236, 255, 226, 255),
            packet(3, 0b00000010),
        ])
        sample = reader.poll()
        np.testing.assert_array_equal(sample.raw_axes, [10, 20, 30, -10, -20, -30])
        self.assertTrue(sample.buttons[1])
        self.assertFalse(sample.buttons[0])
        self.assertTrue(sample.buttons_valid)

    def test_stale_input_and_read_error_fail_closed(self):
        reader, _ = self.make_reader([packet(1, 1, 0, 0, 0, 0, 0)])
        self.assertTrue(reader.poll().valid)
        reader._last_motion_at -= reader.config.stale_timeout_s + 0.01
        self.assertFalse(reader.poll().valid)
        bad_reader, fake = self.make_reader(read_error=OSError("removed"))
        sample = bad_reader.poll()
        self.assertFalse(sample.valid)
        np.testing.assert_array_equal(sample.raw_axes, np.zeros(6))
        self.assertIn("HID read failed", sample.error)
        self.assertTrue(fake.closed)

    def test_buttons_never_enter_motion_vector(self):
        reader, _ = self.make_reader([packet(3, 0b11111111)])
        sample = reader.poll()
        np.testing.assert_array_equal(sample.raw_axes, np.zeros(6))
        self.assertEqual(len(sample.buttons), 8)
        self.assertFalse(sample.valid)
        self.assertTrue(sample.buttons_valid)

    def test_button_freshness_is_independent_from_stale_motion_and_reset(self):
        reader, _ = self.make_reader([packet(3, 0b00000001)])
        sample = reader.poll()
        self.assertFalse(sample.valid)
        self.assertTrue(sample.buttons_valid)
        self.assertTrue(sample.buttons[0])
        reader.reset_input_state()
        reset = reader.poll()
        np.testing.assert_array_equal(reset.raw_axes, np.zeros(6))
        self.assertEqual(reset.buttons, (False,) * 8)
        self.assertFalse(reset.buttons_valid)

    def test_startup_zero_bias_requires_centered_reports_and_is_not_learned_while_moving(self):
        settings = WristConfig(
            raw_ranges=(100.0,) * 6,
            translation_deadzone=0.10,
            rotation_deadzone=0.10,
            zero_bias_calibration_frames=2,
            neutral_startup_s=1.0e-9,
        )
        reports = [
            packet(1, 30, 0, 0, 0, 0, 0),  # outside neutral: cannot arm
            packet(1, 6, 0, 251, 255, 4, 0),
            packet(1, 6, 0, 251, 255, 4, 0),
        ]
        fake = FakeDevice(reports)
        reader = SpaceMouseHid(settings, hid_module=FakeHid(fake))
        reader.open()
        reader._opened_at -= 0.01
        sample = reader.poll()
        self.assertTrue(sample.valid)
        self.assertTrue(sample.neutral_locked)
        self.assertTrue(sample.armed)
        np.testing.assert_array_equal(sample.raw_axes, [6, -5, 4, 0, 0, 0])
        np.testing.assert_allclose(sample.zero_bias_axes, [6, -5, 4, 0, 0, 0])
        np.testing.assert_array_equal(sample.control_axes, np.zeros(6))

        # The captured offset remains fixed after arming; it never follows a
        # later intentional movement.
        fake.reports.append(packet(1, 50, 0, 251, 255, 4, 0))
        moved = reader.poll()
        self.assertTrue(moved.neutral_locked)
        np.testing.assert_allclose(moved.zero_bias_axes, [6, -5, 4, 0, 0, 0])
        self.assertGreater(moved.control_axes[0], 40.0)
