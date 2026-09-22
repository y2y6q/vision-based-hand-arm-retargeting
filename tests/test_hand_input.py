"""Synthetic tests for the extracted camera-finger adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_teleop.hand_input import (
    DEFAULT_ALLEGRO_LOWER_LIMITS,
    DEFAULT_ALLEGRO_UPPER_LIMITS,
    HandCommand,
    HandInputAdapter,
    CameraHandWorker,
    curls_to_allegro_targets,
    estimate_all_finger_curls,
)
from wrist_teleop.runtime_controls import KeyboardCommandRouter


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def synthetic_hand(closed: bool) -> list[Landmark]:
    points = [Landmark(0.0, 0.0) for _ in range(21)]
    for base in (1, 5, 9, 13, 17):
        x = base / 10.0
        points[base] = Landmark(x, 1.0)
        points[base + 1] = Landmark(x, 2.0)
        points[base + 2] = Landmark(x, 2.5)
        points[base + 3] = Landmark(x, 1.5 if closed else 4.0)
    return points


class HandInputTests(unittest.TestCase):
    def test_synthetic_open_and_closed_hands_change_curls(self):
        opened = estimate_all_finger_curls(synthetic_hand(False))
        closed = estimate_all_finger_curls(synthetic_hand(True))
        self.assertTrue(np.all(closed > opened))
        self.assertTrue(np.all((opened >= 0) & (closed <= 1)))

    def test_mapping_shape_order_and_limits(self):
        targets = curls_to_allegro_targets([1, 1, 1, 1], DEFAULT_ALLEGRO_LOWER_LIMITS, DEFAULT_ALLEGRO_UPPER_LIMITS)
        self.assertEqual(targets.shape, (16,))
        self.assertTrue(np.all(targets >= DEFAULT_ALLEGRO_LOWER_LIMITS))
        self.assertTrue(np.all(targets <= DEFAULT_ALLEGRO_UPPER_LIMITS))
        self.assertGreater(targets[1], targets[0])

    def test_adapter_isolated_hold_on_loss_and_stale(self):
        adapter = HandInputAdapter(smoothing_alpha=1.0, stale_timeout_s=0.2)
        tracked = adapter.update_landmarks(synthetic_hand(True), timestamp_s=10.0)
        self.assertTrue(tracked.valid)
        self.assertEqual(tracked.tracking_state, "tracking")
        lost = adapter.mark_lost(timestamp_s=10.1)
        self.assertFalse(lost.valid)
        self.assertEqual(lost.tracking_state, "lost_hold")
        np.testing.assert_allclose(lost.joint_targets, tracked.joint_targets)
        stale = adapter.latest(timestamp_s=11.0)
        self.assertFalse(stale.valid)
        self.assertEqual(stale.calibration_state, "not_required_rule_based_fingers")

    def test_invalid_landmarks_fail_to_hold_without_wrist_data(self):
        adapter = HandInputAdapter()
        command = adapter.update_landmarks([Landmark(0, 0)] * 20, timestamp_s=1.0)
        self.assertFalse(command.valid)
        self.assertEqual(command.joint_targets.shape, (16,))
        self.assertFalse(hasattr(command, "wrist_pose"))

    def test_reset_clears_latest_camera_buffer_without_changing_calibration_mode(self):
        adapter = HandInputAdapter(smoothing_alpha=1.0)
        tracked = adapter.update_landmarks(synthetic_hand(True), timestamp_s=20.0)
        self.assertTrue(tracked.valid)
        reset = adapter.reset(timestamp_s=21.0)
        self.assertFalse(reset.valid)
        self.assertEqual(reset.tracking_state, "waiting_for_hand")
        self.assertEqual(reset.calibration_state, "not_required_rule_based_fingers")
        np.testing.assert_allclose(
            reset.joint_targets,
            np.clip(np.zeros(16), adapter.lower_limits, adapter.upper_limits),
        )

    def test_camera_worker_shutdown_is_cooperative_without_hardware(self):
        class CooperativeWorker(CameraHandWorker):
            def _run(self):
                while not self._stop.is_set():
                    time.sleep(0.001)

        worker = CooperativeWorker(HandInputAdapter(), display=False)
        worker.start()
        self.assertTrue(worker.running)
        worker.close()
        self.assertFalse(worker.running)

    def test_camera_display_routes_keys_and_window_close_without_opening_another_capture(self):
        class Gui:
            WND_PROP_VISIBLE = 0

            def __init__(self):
                self.visible = 1.0
                self.keys = [ord("r")]
                self.frames = 0
                self.text = []

            def imshow(self, _name, _frame):
                self.frames += 1

            def waitKey(self, _delay):
                return self.keys.pop(0) if self.keys else -1

            def getWindowProperty(self, _name, _property):
                return self.visible

            def putText(self, _frame, text, *_args):
                self.text.append(text)

            FONT_HERSHEY_SIMPLEX = 0
            LINE_AA = 0

        router = KeyboardCommandRouter()
        gui = Gui()
        worker = CameraHandWorker(
            HandInputAdapter(),
            display=True,
            keypress_callback=lambda key: router.feed(key, source="camera"),
            status_provider=lambda: {
                "hand_mode": "dex",
                "hand_source": "thumb_index_pinch",
                "active_override": "thumb_index_pinch",
            },
        )
        command = HandCommand(
            joint_targets=np.zeros(16), valid=True, timestamp_s=10.0,
            tracking_state="tracking", calibration_state="dex", curls=np.zeros(4),
            details={"algorithm": "dex-retargeting"},
        )
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        worker._draw_display(gui, frame, command, handedness="Right", timestamp=10.0)
        self.assertTrue(any("detected hand: Right" in line for line in gui.text))
        self.assertTrue(any("DEX ACTIVE" in line for line in gui.text))
        self.assertTrue(any("active: thumb_index_pinch" in line for line in gui.text))
        gui.text.clear()
        held = HandCommand(
            joint_targets=np.zeros(16), valid=True, timestamp_s=10.0,
            tracking_state="lost_hold", calibration_state="dex", curls=np.zeros(4),
            details={"algorithm": "dex-retargeting", "rejected_frame_reason": "no_hand"},
        )
        worker._draw_display(gui, frame, held, handedness=None, timestamp=10.1)
        self.assertTrue(any("DEX HOLD" in line for line in gui.text))
        worker._show_frame(gui, frame)
        self.assertTrue(router.take_reset())
        gui.visible = 0.0
        worker._show_frame(gui, frame)
        worker._show_frame(gui, frame)
        self.assertTrue(router.stop_event.is_set())
        self.assertEqual(router.quit_reason, "camera:Q")
        self.assertEqual(gui.frames, 3)

    def test_camera_worker_selects_the_configured_physical_dex_hand_from_two_labels(self):
        self.assertEqual(
            CameraHandWorker._select_world_hand_index(
                ["Left", "Right"], expected_handedness="Right", input_is_mirrored=False,
            ),
            1,
        )
        self.assertEqual(
            CameraHandWorker._select_world_hand_index(
                ["Left", "Right"], expected_handedness="Right", input_is_mirrored=True,
            ),
            0,
        )

    def test_camera_worker_pairs_selected_dex_hand_with_matching_world_landmarks(self):
        """Right-hand selection must not accidentally send index-zero 3-D points."""

        first_world = [Landmark(-1.0, 0.0, 0.0)]
        second_world = [Landmark(1.0, 0.0, 0.0)]
        index = CameraHandWorker._select_world_hand_index(
            ["Left", "Right"], expected_handedness="Right", input_is_mirrored=False,
        )
        selected = CameraHandWorker._world_landmarks_at(
            [SimpleNamespace(landmark=first_world), SimpleNamespace(landmark=second_world)], index,
        )
        self.assertIs(selected, second_world)
        self.assertIsNone(CameraHandWorker._world_landmarks_at([], index))

    def test_camera_close_releases_the_single_capture_before_joining(self):
        class Capture:
            def __init__(self):
                self.release_calls = 0

            def release(self):
                self.release_calls += 1

        capture = Capture()
        worker = CameraHandWorker(HandInputAdapter(), display=False)
        with worker._capture_lock:
            worker._capture = capture
        worker.close()
        self.assertEqual(capture.release_calls, 1)
        self.assertFalse(worker.status()["capture_open"])
