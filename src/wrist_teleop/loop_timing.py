"""Loop timing: per-stage profiling, a latest-only mailbox, a drift-free fixed-rate clock, and fixed-delay hand resampling.

All durations use ``time.perf_counter``. On Windows, ``time.monotonic`` only
advances in 15.6 ms steps and Python 3.10's ``time.sleep`` rounds up to the
same 15.6 ms timer tick (``sleep(0.020)`` measured 31 ms), so neither can time
or pace a 20 Hz loop. Hand-command timestamps elsewhere stay on
``time.monotonic`` because the camera worker and the adapters share that clock.
"""

from __future__ import annotations

from collections import deque
import contextlib
import os
import threading
import time
from typing import Any, Iterator, Sequence

import numpy as np


class RollingStats:
    """Rolling window of one stage's durations and completion times (thread-safe).

    ``snapshot`` reports how often the stage completes (rate and period
    jitter) and how long it takes (latency mean / p50 / p99 / max).
    """

    def __init__(self, window: int = 600) -> None:
        self._durations: deque[float] = deque(maxlen=int(window))
        self._times: deque[float] = deque(maxlen=int(window))
        self._count = 0
        self._lock = threading.Lock()

    def add(self, duration_s: float, at_s: float | None = None) -> None:
        with self._lock:
            self._durations.append(float(duration_s))
            self._times.append(time.perf_counter() if at_s is None else float(at_s))
            self._count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            durations = np.asarray(self._durations, dtype=np.float64)
            times = np.asarray(self._times, dtype=np.float64)
            count = self._count
        out: dict[str, Any] = {"count": count, "window": int(durations.size)}
        if durations.size:
            ms = 1e3 * durations
            out.update(mean_ms=float(ms.mean()), p50_ms=float(np.percentile(ms, 50)),
                       p99_ms=float(np.percentile(ms, 99)), max_ms=float(ms.max()))
        if times.size >= 3:
            periods = 1e3 * np.diff(times)
            mean_period = float(periods.mean())
            out.update(rate_hz=1e3 / mean_period if mean_period > 0 else None,
                       period_mean_ms=mean_period, period_p99_ms=float(np.percentile(periods, 99)),
                       period_max_ms=float(periods.max()),
                       jitter_p99_ms=float(np.percentile(np.abs(periods - mean_period), 99)))
        return out


class StageProfiler:
    """Named ``RollingStats`` for the stages of one thread or loop."""

    def __init__(self, window: int = 600) -> None:
        self.window = int(window)
        self._stages: dict[str, RollingStats] = {}
        self._lock = threading.Lock()

    def add(self, name: str, duration_s: float, at_s: float | None = None) -> None:
        with self._lock:
            stats = self._stages.get(name)
            if stats is None:
                stats = self._stages[name] = RollingStats(self.window)
        stats.add(duration_s, at_s)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.add(name, end - start, end)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = dict(self._stages)
        return {name: stats.snapshot() for name, stats in stages.items()}


class LatestSlot:
    """Single-item mailbox between a producer thread and a consumer thread.

    ``put`` overwrites whatever is waiting, so the consumer only ever sees the
    newest item and nothing queues up behind a slow consumer; ``dropped``
    counts items that were overwritten before anyone took them. ``take``
    returns an item newer than the last one taken, waiting up to ``timeout``.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any = None
        self._seq = 0
        self._taken_seq = 0
        self._closed = False
        self.published = 0
        self.dropped = 0

    def put(self, item: Any) -> None:
        with self._cond:
            if self._seq > self._taken_seq:
                self.dropped += 1
            self._item = item
            self._seq += 1
            self.published += 1
            self._cond.notify_all()

    def take(self, timeout: float | None = None) -> Any:
        """Newest untaken item, or ``None`` on timeout or after ``close``."""

        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._seq > self._taken_seq, timeout)
            if self._seq <= self._taken_seq:
                return None
            self._taken_seq = self._seq
            return self._item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed


class _WindowsTimerResolution:
    """Request 1 ms timer resolution from ``winmm`` for the life of a loop.

    Without it a Windows ``time.sleep`` cannot wait less than one 15.6 ms
    tick. The request is released in ``close`` (and by Windows at exit).
    """

    def __init__(self) -> None:
        self._winmm: Any = None
        if os.name != "nt":
            return
        try:
            import ctypes
            winmm = ctypes.WinDLL("winmm")
            if winmm.timeBeginPeriod(1) == 0:
                self._winmm = winmm
        except Exception:
            self._winmm = None

    @property
    def active(self) -> bool:
        return self._winmm is not None

    def close(self) -> None:
        winmm, self._winmm = self._winmm, None
        if winmm is not None:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass


class FixedRateClock:
    """Drift-free fixed-rate pacing: ``next_tick += period``, never ``now + period``.

    ``wait`` sleeps to within ``spin_s`` of the next tick and yields the rest,
    so ticks land on a fixed grid regardless of how long each cycle worked.
    If a cycle overran by a whole period or more (a reset, a GC pause), the
    missed ticks are dropped and the grid restarts from now rather than
    bursting several steps back to back; ``overruns`` counts those.
    """

    def __init__(self, rate_hz: float, *, spin_s: float = 0.0015, high_resolution_timer: bool = True) -> None:
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be a positive finite number")
        self.period_s = 1.0 / float(rate_hz)
        self.spin_s = float(spin_s)
        self._timer = _WindowsTimerResolution() if high_resolution_timer else None
        self.next_tick_s = time.perf_counter()
        self.overruns = 0
        self.ticks = 0

    @property
    def high_resolution_timer(self) -> bool:
        return bool(self._timer is not None and self._timer.active)

    def restart(self) -> float:
        """Start a fresh grid at now (after a reset or another long pause)."""

        self.next_tick_s = time.perf_counter()
        return self.next_tick_s

    def wait(self) -> dict[str, float]:
        """Advance one period and wait for it; returns lateness diagnostics in seconds."""

        self.next_tick_s += self.period_s
        now = time.perf_counter()
        if now - self.next_tick_s >= self.period_s:
            self.overruns += 1
            missed = now - self.next_tick_s
            self.next_tick_s = now
            self.ticks += 1
            return {"late_s": missed, "slept_s": 0.0, "dropped": True}
        start = now
        remaining = self.next_tick_s - now
        if remaining > self.spin_s:
            time.sleep(remaining - self.spin_s)
        while time.perf_counter() < self.next_tick_s:
            time.sleep(0)
        woke = time.perf_counter()
        self.ticks += 1
        return {"late_s": woke - self.next_tick_s, "slept_s": woke - start, "dropped": False}

    def close(self) -> None:
        if self._timer is not None:
            self._timer.close()


class FixedDelayResampler:
    """Camera hand targets read off at a fixed delay behind each simulator tick.

    The camera delivers frames every 32 or 48 ms and each takes a slightly
    different time to process, while the simulator ticks every 50 ms; taking
    "the newest result" at each tick therefore sees the hand 0-46 ms late,
    a different amount every tick. Here every camera sample is kept with the
    time its frame was read (``time.perf_counter``), and the tick at ``now``
    uses the hand as it was at exactly ``now - delay_s``: linearly
    interpolated between the two samples on either side of that moment. Every
    tick then sees the hand the same ``delay_s`` ago, and evenly spaced ticks
    see evenly spaced moments of the hand.

    If the sample after ``now - delay_s`` has not arrived yet (a late frame),
    the newest sample is held - never extrapolated - and the tick is counted
    as ``held_newest``. Until the first sample is ``delay_s`` old, the oldest
    sample is used (``before_oldest``). The output is always a mix of two
    consecutive camera samples; nothing else is blended in. ``delay_s = 0``
    simply returns the newest sample.

    Hand validity (hold / safe-open) stays with the caller, undelayed.
    """

    STATES = ("interpolated", "held_newest", "before_oldest")

    def __init__(self, size: int, delay_s: float, *, capacity: int = 64) -> None:
        if size <= 0 or capacity < 2 or not np.isfinite(delay_s) or delay_s < 0.0:
            raise ValueError("FixedDelayResampler needs a positive size, capacity >= 2 and a delay >= 0")
        self.size, self.delay_s, self.capacity = int(size), float(delay_s), int(capacity)
        self.counts = {state: 0 for state in self.STATES}
        self.reset()

    def reset(self, discard_until_s: float | None = None) -> None:
        """Forget all samples; with ``discard_until_s``, also ignore any older frame pushed later."""

        self._times: deque[float] = deque(maxlen=self.capacity)
        self._targets: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._floor_s = None if discard_until_s is None else float(discard_until_s)
        self._last: dict[str, Any] = {"state": "empty"}

    def __len__(self) -> int:
        return len(self._times)

    def push(self, frame_time_s: float, target: Sequence[float]) -> bool:
        """Add one camera sample; out-of-order, repeated or discarded frames are ignored."""

        values = np.asarray(target, dtype=np.float64).reshape(-1)
        if values.shape != (self.size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"resampled target must be {self.size} finite values")
        time_s = float(frame_time_s)
        if not np.isfinite(time_s):
            return False
        if self._floor_s is not None and time_s <= self._floor_s:
            return False
        if self._times and time_s <= self._times[-1]:
            return False
        self._times.append(time_s)
        self._targets.append(values.copy())
        return True

    def sample(self, now_s: float) -> np.ndarray | None:
        """Target at ``now_s - delay_s``, or ``None`` before any sample arrived."""

        if not self._times:
            self._last = {"state": "empty"}
            return None
        query = float(now_s) - self.delay_s
        times, targets = self._times, self._targets
        if query >= times[-1]:
            state, output = "held_newest", targets[-1]
        elif query < times[0]:
            state, output = "before_oldest", targets[0]
        else:
            # Ticks only move forward, so samples before the pair around the
            # query are never needed again.
            while len(times) >= 2 and times[1] <= query:
                times.popleft()
                targets.popleft()
            span = times[1] - times[0]
            weight = (query - times[0]) / span
            state, output = "interpolated", targets[0] + weight * (targets[1] - targets[0])
        self.counts[state] += 1
        self._last = {
            "state": state,
            # How far the newest sample is ahead of the moment shown; negative
            # means the camera was late and the newest sample is being held.
            "newest_lead_ms": 1e3 * (times[-1] - query),
            "samples_buffered": len(times),
        }
        return np.array(output, dtype=np.float64, copy=True)

    def describe(self) -> dict[str, Any]:
        """Last tick's state plus cumulative tick counts per state."""

        return {"delay_ms": 1e3 * self.delay_s, **self._last, "counts": dict(self.counts)}
