"""Unit tests for loop timing: stats, latest-only slot, fixed-rate clock, fixed-delay hand resampling."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_teleop.loop_timing import (  # noqa: E402
    FixedDelayResampler,
    FixedRateClock,
    LatestSlot,
    RollingStats,
    StageProfiler,
)


class RollingStatsTest(unittest.TestCase):
    def test_rate_latency_and_jitter_from_explicit_times(self) -> None:
        stats = RollingStats(window=100)
        for index in range(11):
            # 50 ms period, durations 1..11 ms.
            stats.add((index + 1) * 1e-3, at_s=index * 0.05)
        snap = stats.snapshot()
        self.assertEqual(snap["count"], 11)
        self.assertAlmostEqual(snap["rate_hz"], 20.0, places=6)
        self.assertAlmostEqual(snap["mean_ms"], 6.0, places=6)
        self.assertAlmostEqual(snap["max_ms"], 11.0, places=6)
        self.assertAlmostEqual(snap["jitter_p99_ms"], 0.0, places=6)

    def test_window_bounds_memory_but_count_keeps_total(self) -> None:
        stats = RollingStats(window=5)
        for index in range(20):
            stats.add(0.001, at_s=float(index))
        snap = stats.snapshot()
        self.assertEqual((snap["count"], snap["window"]), (20, 5))

    def test_profiler_stage_context_records(self) -> None:
        profiler = StageProfiler()
        with profiler.stage("work"):
            pass
        self.assertEqual(profiler.snapshot()["work"]["count"], 1)


class LatestSlotTest(unittest.TestCase):
    def test_only_newest_item_is_taken_and_overwrites_are_counted(self) -> None:
        slot = LatestSlot()
        for item in range(5):
            slot.put(item)
        self.assertEqual(slot.take(timeout=0.0), 4)
        self.assertEqual(slot.dropped, 4)
        # Nothing new: an item is never handed out twice.
        self.assertIsNone(slot.take(timeout=0.0))
        slot.put(5)
        self.assertEqual(slot.take(timeout=0.0), 5)
        self.assertEqual((slot.published, slot.dropped), (6, 4))

    def test_take_wakes_on_put_and_on_close(self) -> None:
        slot = LatestSlot()
        threading.Timer(0.02, slot.put, args=("frame",)).start()
        self.assertEqual(slot.take(timeout=2.0), "frame")
        threading.Timer(0.02, slot.close).start()
        start = time.perf_counter()
        self.assertIsNone(slot.take(timeout=2.0))
        self.assertLess(time.perf_counter() - start, 1.0)
        self.assertTrue(slot.closed)


class FixedRateClockTest(unittest.TestCase):
    def test_grid_does_not_drift_with_variable_work(self) -> None:
        clock = FixedRateClock(50.0)  # 20 ms period keeps the test short
        try:
            start = clock.restart()
            rng = np.random.default_rng(0)
            ticks = 40
            for _ in range(ticks):
                time.sleep(float(rng.uniform(0.0, 0.012)))  # work shorter than a period
                clock.wait()
            elapsed = time.perf_counter() - start
            # Ticks stay on start + k * period: the total does not grow with
            # the per-cycle work (sleep-to-rate drift would add it up).
            self.assertAlmostEqual(clock.next_tick_s, start + ticks * clock.period_s, places=9)
            self.assertLess(abs(elapsed - ticks * clock.period_s), 0.03)
            self.assertEqual(clock.overruns, 0)
        finally:
            clock.close()

    def test_short_overrun_catches_up_on_the_same_grid(self) -> None:
        clock = FixedRateClock(50.0)
        try:
            start = clock.restart()
            time.sleep(0.03)  # 1.5 periods of work: late, but under two periods
            result = clock.wait()
            self.assertFalse(result["dropped"])
            self.assertEqual(clock.overruns, 0)
            clock.wait()
            self.assertAlmostEqual(clock.next_tick_s, start + 2 * clock.period_s, places=9)
        finally:
            clock.close()

    def test_long_overrun_drops_missed_ticks_instead_of_bursting(self) -> None:
        clock = FixedRateClock(50.0)
        try:
            clock.restart()
            time.sleep(0.1)  # five periods
            result = clock.wait()
            self.assertTrue(result["dropped"])
            self.assertEqual(clock.overruns, 1)
            # The grid restarted at "now", so the next wait is a full period.
            begin = time.perf_counter()
            clock.wait()
            self.assertGreater(time.perf_counter() - begin, 0.5 * clock.period_s)
        finally:
            clock.close()

    def test_rejects_invalid_rate(self) -> None:
        for rate in (0.0, -5.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FixedRateClock(rate)


def _camera_frames(seconds: float, seed: int = 0) -> list[tuple[float, float]]:
    """(read time, ready time) of the 30 fps webcam as measured here: frames 32
    or 48 ms apart (a 16 ms delivery grid) and 8-15 ms of MediaPipe processing,
    so the worst case 48 + 15 ms stays under the 65 ms delay."""

    rng = np.random.default_rng(seed)
    frames, read = [], 0.0
    while read < seconds:
        read += 0.048 if rng.random() < 0.08 else 0.032
        frames.append((read, read + float(rng.uniform(0.008, 0.015))))
    return frames


class FixedDelayResamplerTest(unittest.TestCase):
    TICK = 0.05
    DELAY = 0.065

    def run_ticks(self, resampler: FixedDelayResampler, frames, ticks: int, start: float = 0.5):
        """Feed each frame once it is ready; the hand signal equals its frame time."""

        outputs, pending = [], list(frames)
        for index in range(ticks):
            now = start + index * self.TICK
            while pending and pending[0][1] <= now:
                read, _ready = pending.pop(0)
                resampler.push(read, [read, -read])
            outputs.append((now, resampler.sample(now), resampler.describe()["state"]))
        return outputs

    def test_every_tick_sees_the_hand_exactly_delay_ago(self) -> None:
        resampler = FixedDelayResampler(2, self.DELAY)
        outputs = self.run_ticks(resampler, _camera_frames(12.0), ticks=200)
        states = [state for _now, _out, state in outputs]
        self.assertEqual(states.count("interpolated"), len(states))
        for now, out, _state in outputs:
            # A hand moving linearly in time is reproduced exactly at now - delay.
            np.testing.assert_allclose(out, [now - self.DELAY, -(now - self.DELAY)], atol=1e-12)
        # So evenly spaced ticks see evenly spaced moments of the hand.
        shown = np.array([out[0] for _now, out, _state in outputs])
        np.testing.assert_allclose(np.diff(shown), self.TICK, atol=1e-12)

    def test_newest_sample_alone_would_be_uneven(self) -> None:
        # The behaviour being replaced: the newest ready sample's age varies by
        # tens of milliseconds from tick to tick.
        frames = _camera_frames(12.0)
        ages = []
        for index in range(200):
            now = 0.5 + index * self.TICK
            ages.append(now - max(read for read, ready in frames if ready <= now))
        self.assertGreater(max(ages) - min(ages), 0.03)

    def test_late_frame_holds_the_newest_sample_without_extrapolating(self) -> None:
        resampler = FixedDelayResampler(1, self.DELAY)
        resampler.push(1.000, [0.0])
        resampler.push(1.032, [1.0])
        np.testing.assert_allclose(resampler.sample(1.032 + self.DELAY + 0.04), [1.0])
        state = resampler.describe()
        self.assertEqual(state["state"], "held_newest")
        self.assertAlmostEqual(state["newest_lead_ms"], -40.0, places=6)
        self.assertEqual(state["counts"]["held_newest"], 1)

    def test_oldest_sample_until_it_is_delay_old_and_none_before_any(self) -> None:
        resampler = FixedDelayResampler(1, self.DELAY)
        self.assertIsNone(resampler.sample(1.0))
        resampler.push(1.0, [3.0])
        np.testing.assert_allclose(resampler.sample(1.01), [3.0])
        self.assertEqual(resampler.describe()["state"], "before_oldest")

    def test_output_is_a_mix_of_two_neighbouring_samples(self) -> None:
        rng = np.random.default_rng(1)
        resampler = FixedDelayResampler(3, self.DELAY)
        frames = _camera_frames(3.0, seed=2)
        values = {read: rng.uniform(-1.0, 1.0, size=3) for read, _ready in frames}
        reads = sorted(values)
        pending = list(reads)
        for now in np.arange(0.2, 3.0, 0.01):
            while pending and pending[0] <= now:
                read = pending.pop(0)
                resampler.push(read, values[read])
            out = resampler.sample(now)
            query = now - self.DELAY
            before = max(r for r in reads if r <= query)
            after = min(r for r in reads if r > query)
            low = np.minimum(values[before], values[after]) - 1e-12
            high = np.maximum(values[before], values[after]) + 1e-12
            self.assertTrue(np.all(out >= low) and np.all(out <= high))

    def test_zero_delay_returns_the_newest_sample(self) -> None:
        resampler = FixedDelayResampler(1, 0.0)
        resampler.push(1.0, [1.0])
        resampler.push(1.032, [2.0])
        np.testing.assert_allclose(resampler.sample(1.04), [2.0])

    def test_reset_drops_samples_and_frames_read_before_it(self) -> None:
        resampler = FixedDelayResampler(1, self.DELAY)
        resampler.push(1.0, [1.0])
        resampler.reset(discard_until_s=1.1)
        self.assertEqual(len(resampler), 0)
        self.assertFalse(resampler.push(1.05, [5.0]))  # read before the reset, processed after
        self.assertTrue(resampler.push(1.12, [2.0]))
        np.testing.assert_allclose(resampler.sample(1.15), [2.0])

    def test_out_of_order_or_repeated_frames_are_ignored(self) -> None:
        resampler = FixedDelayResampler(1, self.DELAY)
        self.assertTrue(resampler.push(1.0, [1.0]))
        self.assertFalse(resampler.push(1.0, [9.0]))
        self.assertFalse(resampler.push(0.9, [9.0]))
        self.assertFalse(resampler.push(float("nan"), [9.0]))
        self.assertEqual(len(resampler), 1)

    def test_rejects_bad_targets_and_settings(self) -> None:
        resampler = FixedDelayResampler(16, self.DELAY)
        with self.assertRaises(ValueError):
            resampler.push(1.0, np.zeros(15))
        with self.assertRaises(ValueError):
            resampler.push(1.0, np.full(16, np.nan))
        for size, delay in ((0, 0.05), (16, -0.01), (16, float("nan"))):
            with self.assertRaises(ValueError):
                FixedDelayResampler(size, delay)


if __name__ == "__main__":
    unittest.main()
