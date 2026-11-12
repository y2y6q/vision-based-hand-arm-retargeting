"""R during a slow (LIBERO hard) reset: one R = one reset, and the reset is shown on screen.

    .\\.venv-libero\\Scripts\\python.exe -m pytest -q .\\tests\\test_reset_feedback.py

The router and notice tests run in either interpreter; the entry test needs
.venv-libero (robosuite 1.4) and builds the real LIBERO task (~20 s).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wrist_teleop.runtime_controls import (  # noqa: E402
    RESET_NOTICE,
    RESET_NOTICE_WAIT_S,
    KeyboardCommandRouter,
    ResetFeedback,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import robosuite

    LEGACY_ROBOSUITE = robosuite.__version__.startswith("1.4")
except ImportError:  # pragma: no cover
    LEGACY_ROBOSUITE = False


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class ResetHoldTests(unittest.TestCase):
    def test_r_while_a_reset_runs_is_dropped_counted_and_accepted_after_the_grace(self):
        clock = FakeClock(1.0)
        router = KeyboardCommandRouter(repeat_release_s=0.25, clock=clock)
        self.assertEqual(router.feed("r", timestamp_s=1.0), "reset")
        self.assertTrue(router.take_reset())
        router.hold_resets()
        self.assertTrue(router.resets_held)
        # The picture is frozen: the operator presses R again, twice.
        self.assertIsNone(router.feed("r", timestamp_s=1.6))
        self.assertIsNone(router.feed(ord("R"), timestamp_s=2.3))
        self.assertFalse(router.take_reset())
        clock.now = 3.4  # first step on the new model done
        self.assertEqual(router.release_resets(grace_s=0.5), 2)
        self.assertFalse(router.resets_held)
        self.assertIsNone(router.feed("r", timestamp_s=3.7))  # inside the grace
        self.assertFalse(router.take_reset())
        self.assertEqual(router.feed("r", timestamp_s=4.2), "reset")  # a fresh press after it
        self.assertTrue(router.take_reset())

    def test_q_still_quits_while_resets_are_held(self):
        router = KeyboardCommandRouter()
        router.hold_resets()
        self.assertEqual(router.feed("q", source="views", timestamp_s=5.0), "quit")
        self.assertTrue(router.stop_event.is_set())
        self.assertEqual(router.quit_reason, "views:Q")

    def test_holding_r_down_through_the_end_of_a_reset_does_not_reset_again(self):
        clock = FakeClock(0.0)
        router = KeyboardCommandRouter(repeat_release_s=0.25, clock=clock)
        router.hold_resets()
        results = []
        for k in range(100):  # auto-repeat every 30 ms for 3 s; the hold ends at 1 s
            t = 0.03 * k
            if t >= 1.0 and router.resets_held:
                clock.now = t
                router.release_resets(grace_s=0.5)
            results.append(router.feed("r", timestamp_s=t))
        self.assertTrue(all(result is None for result in results))
        self.assertFalse(router.take_reset())
        self.assertEqual(router.feed("r", timestamp_s=3.5), "reset")  # key quiet, then pressed again

    def test_without_a_hold_r_behaves_as_before(self):
        router = KeyboardCommandRouter(repeat_release_s=0.25)
        self.assertEqual(router.feed("r", timestamp_s=1.0), "reset")
        self.assertIsNone(router.feed("r", timestamp_s=1.1))
        self.assertTrue(router.take_reset())
        self.assertEqual(router.feed("r", timestamp_s=1.4), "reset")
        self.assertTrue(router.take_reset())
        self.assertFalse(router.resets_held)

    def test_a_pending_r_is_dropped_by_the_hold(self):
        # An R that slipped in between take_reset and hold_resets belongs to the reset being started.
        router = KeyboardCommandRouter()
        router.feed("r", timestamp_s=1.0)
        self.assertTrue(router.take_reset())
        self.assertEqual(router.feed("r", timestamp_s=1.5), "reset")
        router.hold_resets()
        self.assertFalse(router.take_reset())


class FakeView:
    def __init__(self) -> None:
        self.calls: list[tuple[object, float]] = []

    def show_notice(self, text, wait_s=0.0):
        self.calls.append((text, wait_s))
        return False


class ResetFeedbackTests(unittest.TestCase):
    def test_begin_holds_r_and_shows_the_notice_settle_releases_and_clears_it(self):
        router = KeyboardCommandRouter()
        feedback = ResetFeedback(router, grace_s=0.0)
        view = FakeView()
        feedback.begin(view)
        self.assertTrue(feedback.active)
        self.assertTrue(router.resets_held)
        self.assertEqual(view.calls, [(RESET_NOTICE, RESET_NOTICE_WAIT_S)])
        self.assertIsNone(router.feed("r"))
        record = feedback.settle(view)
        self.assertEqual(view.calls[-1], (None, 0.0))
        self.assertEqual(record["event"], "reset_settled")
        self.assertEqual(record["what"], "reset")
        self.assertEqual(record["ignored_r_presses"], 1)
        self.assertGreaterEqual(record["held_s"], 0.0)
        self.assertFalse(feedback.active)
        self.assertFalse(router.resets_held)
        self.assertIsNone(feedback.settle(view))  # nothing held any more

    def test_works_without_a_view_and_with_a_custom_label(self):
        router = KeyboardCommandRouter()
        feedback = ResetFeedback(router)
        feedback.begin(None, label="hand mount switch")
        self.assertTrue(router.resets_held)
        record = feedback.settle(object())  # a view without show_notice
        self.assertEqual(record["what"], "hand mount switch")
        self.assertEqual(record["r_grace_s"], 0.5)
        self.assertFalse(router.resets_held)


class _RecordingCv2:
    """Real OpenCV drawing; imshow only records (no window in tests)."""

    def __init__(self) -> None:
        self.shown: list[tuple[str, np.ndarray]] = []

    def imshow(self, name, image):
        self.shown.append((name, image.copy()))

    def __getattr__(self, name):
        return getattr(cv2, name)


@unittest.skipIf(cv2 is None, "needs OpenCV")
class NoticeDisplayTests(unittest.TestCase):
    def _window(self):
        from wrist_teleop.config import TriViewConfig
        from wrist_teleop.libero_views import FollowCamera, LiberoViewWindow, Panel

        follow = FollowCamera(azimuth_deg=90.0, elevation_deg=-20.0, distance_m=0.4)
        panels = (Panel("front", "", camera="frontview"), Panel("wrist", "a", follow=follow),
                  Panel("side", "b", follow=follow))
        return LiberoViewWindow(TriViewConfig(), panels, size=(960, 540), show_window=True)

    @staticmethod
    def _amber(image: np.ndarray) -> np.ndarray:
        return (image[..., 2] > 200) & (image[..., 0] < 80)  # BGR: red high, blue low

    def test_notice_is_drawn_inside_the_main_view_only(self):
        from wrist_teleop.libero_views import draw_notice, layout

        rects = layout(960, 540)
        image = np.full((540, 960, 3), 200, np.uint8)
        draw_notice(cv2, image, RESET_NOTICE, rects["front"], 0.5)
        x, y, w, h = rects["front"]
        changed = np.any(image != 200, axis=2)
        self.assertTrue(changed[y:y + h, x:x + w].any())
        self.assertFalse(changed[:, x + w:].any())  # the two right panels are untouched
        self.assertTrue(self._amber(image).any())

    def test_long_notice_shrinks_to_a_narrow_view(self):
        from wrist_teleop.libero_views import draw_notice

        image = np.full((300, 900, 3), 200, np.uint8)
        draw_notice(cv2, image, RESET_NOTICE, (0, 0, 240, 300), 1.0)
        self.assertTrue(self._amber(image[:, :240]).any())
        self.assertFalse(np.any(image[:, 240:] != 200))

    def test_show_puts_the_notice_on_the_displayed_copy_only(self):
        window = self._window()
        recorder = _RecordingCv2()
        frame = np.full((540, 960, 3), 120, np.uint8)
        self.assertFalse(window.show_notice("RESETTING"))  # no render thread: returns at once
        window._show(recorder, frame)
        name, shown = recorder.shown[-1]
        self.assertEqual(name, window.config.window_name)
        self.assertTrue(self._amber(shown).any())
        self.assertTrue(np.all(frame == 120))  # last_frame stays clean
        self.assertEqual(window._notice_shown, window._notice_version)
        window.show_notice(None)
        window._show(recorder, frame)
        self.assertTrue(np.all(recorder.shown[-1][1] == 120))

    def test_show_notice_waits_until_the_render_thread_has_shown_it(self):
        window = self._window()
        window._thread = object()  # stands in for a running render thread
        recorder = _RecordingCv2()
        frame = np.zeros((540, 960, 3), np.uint8)
        painter = threading.Timer(0.05, lambda: window._show(recorder, frame))
        painter.start()
        started = time.perf_counter()
        self.assertTrue(window.show_notice("RESETTING", wait_s=2.0))
        self.assertLess(time.perf_counter() - started, 1.0)
        painter.join()
        started = time.perf_counter()  # nobody draws: the wait gives up after wait_s
        self.assertFalse(window.show_notice(None, wait_s=0.1))
        self.assertLess(time.perf_counter() - started, 0.5)


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class LiberoEntryResetTests(unittest.TestCase):
    def test_r_pressed_during_a_reset_does_not_reset_again(self):
        from wrist_teleop import libero_spatial_teleop as L
        from wrist_teleop.config import WristConfig
        from wrist_teleop.libero_views import LiberoViewWindow

        router = KeyboardCommandRouter()
        router.feed("r", source="test")  # the loop resets on its first pass
        original_reset = L.OfficialLiberoSpatialSession.reset
        presses: list[object] = []

        def reset_with_impatient_operator(session):
            result = original_reset(session)
            if router.resets_held:  # the picture is frozen: R twice more (the 2026-11-07 run)
                now = time.monotonic()
                presses.append(router.feed("r", source="test", timestamp_s=now))
                presses.append(router.feed("r", source="test", timestamp_s=now + 0.3))
            return result

        notices: list[object] = []
        original_show = LiberoViewWindow.show_notice

        def record_notice(window, text, wait_s=0.0):
            notices.append(text)
            return original_show(window, text, wait_s)

        before = time.time()
        with patch.object(L, "KeyboardCommandRouter", lambda: router), \
                patch.object(L.OfficialLiberoSpatialSession, "reset", reset_with_impatient_operator), \
                patch.object(LiberoViewWindow, "show_notice", record_notice):
            code = L.run_libero_spatial_teleop(
                L.PANDA_SHADOW_SPEC, WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json"),
                task_id=0, task_order=0, init_state_id=0, seed=7, headless=True, duration_s=5.0,
                camera_index=0, camera_display=False, camera_mirror=True, hand_control="off",
                synthetic_input=True, three_view=True, suite="libero_object", view_size=(960, 540),
            )
        self.assertEqual(code, 0)
        self.assertEqual(presses, [None, None])
        runs = [p for p in (ROOT / "outputs" / "libero").glob("*_panda_shadow_teleop")
                if p.stat().st_mtime >= before - 1.0]
        self.assertTrue(runs)
        events = [json.loads(line) for line in (max(runs) / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds.count("environment_reset"), 1)
        settled = [e for e in events if e["event"] == "reset_settled"]
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]["ignored_r_presses"], 2)
        self.assertGreater(kinds.index("reset_settled"), kinds.index("environment_reset"))
        self.assertIn("step", kinds[kinds.index("reset_settled"):])  # control went on afterwards
        self.assertEqual(notices, [RESET_NOTICE, None])


if __name__ == "__main__":
    unittest.main()
