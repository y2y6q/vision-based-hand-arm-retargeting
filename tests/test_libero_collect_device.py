"""The collection entry's SpaceMouse after a layout reset (hardware-free, timed fake HID).

The fake device behaves like the SpaceMouse Wireless: nothing while the cap
rests, a report every 10 ms while it is pushed (translation report 1 and
rotation report 2), a last zero report when it comes back. The loop polls at
20 Hz and maps every sample the way the collection loop does.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wrist_teleop.config import WristConfig  # noqa: E402
from wrist_teleop.device import SpaceMouseHid  # noqa: E402
from wrist_teleop.libero_collect_device import KeepNeutralSpaceMouse  # noqa: E402
from wrist_teleop.mapper import SixDofMapper  # noqa: E402

REPORT_S = 0.010  # one HID report per 10 ms while the cap is off centre
TICK_S = 0.050  # the 20 Hz loop


def _le16(value: int) -> list:
    value = int(value) & 0xFFFF
    return [value & 0xFF, value >> 8]


def _reports(axes) -> list:
    tx, ty, tz, rx, ry, rz = (int(round(v)) for v in axes)
    return [[1, *_le16(tx), *_le16(ty), *_le16(tz)], [2, *_le16(rx), *_le16(ry), *_le16(rz)]]


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


class TimedDevice:
    """HID reports that become readable at their scheduled time (like the OS buffer)."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.schedule: list = []  # (time, report), in time order

    def at(self, t: float, axes) -> None:
        for report in _reports(axes):
            self.schedule.append((t, report))
        self.schedule.sort(key=lambda item: item[0])

    def hold(self, start: float, end: float, axes) -> None:
        """The cap held at ``axes`` from start to end (a report every 10 ms)."""
        t = start
        while t < end - 1e-9:
            self.at(t, axes)
            t += REPORT_S

    def ramp(self, start: float, duration: float, begin, end) -> float:
        """The cap moved from ``begin`` to ``end`` over ``duration``; returns the end time."""
        steps = max(int(round(duration / REPORT_S)), 1)
        begin, end = np.asarray(begin, dtype=float), np.asarray(end, dtype=float)
        for i in range(1, steps + 1):
            self.at(start + i * REPORT_S, begin + (end - begin) * i / steps)
        return start + steps * REPORT_S

    def push(self, start: float, axes, hold_s: float = 0.6) -> float:
        """A brisk push: 50 ms up, held, 50 ms back, a last zero report; returns the release time."""
        up = self.ramp(start, 0.05, np.zeros(6), axes)
        self.hold(up, up + hold_s, axes)
        down = self.ramp(up + hold_s, 0.05, axes, np.zeros(6))
        self.at(down, np.zeros(6))
        return down

    # hidapi device API
    def open_path(self, path) -> None:
        self.path = path

    def set_nonblocking(self, value) -> None:
        self.nonblocking = value

    def read(self, _size):
        if self.schedule and self.schedule[0][0] <= self.clock.now + 1e-9:
            return self.schedule.pop(0)[1]
        return []

    def close(self) -> None:
        self.closed = True


class FakeHid:
    def __init__(self, device) -> None:
        self.device_instance = device

    @staticmethod
    def enumerate():
        return [{"path": r"\\?\HID#VID_256F&PID_C62E#fake", "vendor_id": 0x256F, "product_id": 0xC62E,
                 "product_string": "SpaceMouse Wireless"}]

    def device(self):
        return self.device_instance


CONFIG = WristConfig.load(ROOT / "configs" / "spacemouse_wrist.json")
PUSH = (0.5 * CONFIG.raw_ranges[0], 0, 0, 0, 0, 0)  # half way forward: far past the 8 % deadzone
BIAS = (6, -4, 3, 0, 2, -1)  # what this cap reports at rest (inside the deadzone)


class _Rig:
    """One device on a fake clock, polled and mapped at 20 Hz like the collection loop."""

    def __init__(self, reader_class) -> None:
        self.clock = Clock()
        self.device = TimedDevice(self.clock)
        self.patch = mock.patch("wrist_teleop.device.time", SimpleNamespace(monotonic=self.clock.monotonic))
        self.patch.start()
        self.reader = reader_class(CONFIG, hid_module=FakeHid(self.device))
        self.mapper = SixDofMapper(CONFIG)
        self.reader.open()

    def stop(self) -> None:
        self.patch.stop()

    def calibrate_at_startup(self) -> None:
        """The operator's hand rests on the cap at startup: quiet reports, the neutral locks."""
        self.device.hold(self.clock.now, self.clock.now + 0.2, BIAS)
        self.run(0.5)
        assert self.reader.poll().neutral_locked

    def reset(self) -> None:
        """A new layout (rebind): mapper and device inputs back to neutral."""
        self.mapper.reset()
        self.reader.reset_input_state()

    def run(self, seconds: float) -> list:
        """Poll + map at 20 Hz; returns (time, command norm) per tick."""
        out = []
        end = self.clock.now + seconds
        while self.clock.now < end - 1e-9:
            self.clock.now += TICK_S
            sample = self.reader.poll()
            command = self.mapper.map(sample.control_axes, valid=bool(sample.valid and sample.armed))
            out.append((self.clock.now, float(np.linalg.norm(command))))
        return out

    @staticmethod
    def first_motion(trace: list):
        return next((t for t, norm in trace if norm > 0.0), None)


class KeepNeutralSpaceMouseTests(unittest.TestCase):
    def rig(self, reader_class=KeepNeutralSpaceMouse) -> _Rig:
        rig = _Rig(reader_class)
        self.addCleanup(rig.stop)
        return rig

    def test_old_reader_ignores_brisk_pushes_after_a_reset(self):
        """What the operator saw: after a reset the plain reader needs 8 quiet reports first."""
        rig = self.rig(SpaceMouseHid)
        rig.calibrate_at_startup()
        rig.reset()
        rig.run(2.0)  # the planner drives the hand over the target; the cap rests (no reports)
        t = rig.clock.now
        for _ in range(4):  # four brisk pushes, 0.3 s apart
            t = rig.device.push(t + 0.3, PUSH) + 0.0
        trace = rig.run(t - rig.clock.now + 0.5)
        self.assertIsNone(rig.first_motion(trace))
        self.assertFalse(rig.reader.poll().neutral_locked)

    def test_control_comes_back_with_the_first_push_after_a_reset(self):
        rig = self.rig()
        rig.calibrate_at_startup()
        bias = rig.reader.poll().zero_bias_axes.copy()
        rig.reset()
        self.assertEqual(rig.reader.kept_neutral_resets, 1)
        np.testing.assert_array_equal(rig.reader.poll().zero_bias_axes, bias)
        rig.run(2.0)  # cap untouched through the planner move
        push_at = rig.clock.now + 0.02
        rig.device.push(push_at, PUSH)
        trace = rig.run(0.5)
        moved = rig.first_motion(trace)
        self.assertIsNotNone(moved)
        self.assertLessEqual(moved - push_at, 2 * TICK_S + 1e-9)  # the next tick or the one after

    def test_arms_0_3_s_after_the_reset_at_the_earliest(self):
        rig = self.rig()
        rig.calibrate_at_startup()
        reset_at = rig.clock.now
        rig.reset()
        rig.device.push(reset_at + 0.01, PUSH, hold_s=1.0)  # pushed right away: ignored until it is let go
        trace = rig.run(0.25)
        self.assertIsNone(rig.first_motion(trace))

    def test_a_cap_pushed_through_the_reset_moves_nothing_until_let_go(self):
        rig = self.rig()
        rig.calibrate_at_startup()
        start = rig.clock.now
        rig.device.hold(start, start + 2.0, PUSH)  # held forward before, during and after the reset
        rig.run(0.3)
        rig.reset()
        trace = rig.run(1.5)  # still held: no motion in the new layout
        self.assertIsNone(rig.first_motion(trace))
        release = rig.device.ramp(start + 2.0, 0.05, PUSH, np.zeros(6))
        rig.device.at(release, np.zeros(6))
        rig.run(release - rig.clock.now + 0.4)
        push_at = rig.clock.now + 0.02
        rig.device.push(push_at, PUSH)
        moved = rig.first_motion(rig.run(0.5))
        self.assertIsNotNone(moved)  # after the release the next push drives the arm
        self.assertLessEqual(moved - push_at, 2 * TICK_S + 1e-9)

    def test_reports_from_before_the_reset_are_not_replayed(self):
        rig = self.rig()
        rig.calibrate_at_startup()
        t = rig.clock.now
        rig.device.push(t + 0.01, PUSH, hold_s=0.3)  # pushed and let go while the scene was rebuilt (no polls)
        rig.clock.now = t + 1.5
        rig.reset()
        trace = rig.run(1.0)
        self.assertIsNone(rig.first_motion(trace))

    def test_a_device_without_a_startup_neutral_calibrates_as_before(self):
        rig = self.rig()
        rig.reset()  # no quiet reports yet: nothing to keep
        self.assertEqual(rig.reader.kept_neutral_resets, 0)
        self.assertFalse(rig.reader.poll().neutral_locked)
        t = rig.device.push(rig.clock.now + 0.1, PUSH)
        self.assertIsNone(rig.first_motion(rig.run(t - rig.clock.now + 0.3)))
        rig.device.hold(rig.clock.now, rig.clock.now + 0.2, BIAS)  # the hand rests on it: quiet reports
        rig.run(0.6)
        self.assertTrue(rig.reader.poll().neutral_locked)
        push_at = rig.clock.now + 0.02
        rig.device.push(push_at, PUSH)
        self.assertIsNotNone(rig.first_motion(rig.run(0.5)))

    def test_closed_and_opened_again_calibrates_afresh(self):
        rig = self.rig()
        rig.calibrate_at_startup()
        rig.reader.close()
        rig.reset()
        self.assertEqual(rig.reader.kept_neutral_resets, 0)
        rig.reader.open()
        sample = rig.reader.poll()
        self.assertFalse(sample.neutral_locked)
        self.assertFalse(sample.armed)
        np.testing.assert_array_equal(sample.zero_bias_axes, np.zeros(6))


try:
    import robosuite

    LEGACY_ROBOSUITE = robosuite.__version__.startswith("1.4")
except ImportError:  # pragma: no cover
    LEGACY_ROBOSUITE = False


class LiveDevice(TimedDevice):
    """Fake HID for the real collection loop: the step hook says what the cap does on the next tick."""

    def __init__(self) -> None:
        super().__init__(Clock())
        self.pending: list = []
        self.axes = None  # None: the cap rests and sends nothing

    def set(self, axes) -> None:
        if axes is None:
            if self.axes is not None:  # let go: one last zero report, then silence
                self.pending.extend(_reports(np.zeros(6)))
            self.axes = None
            return
        self.axes = axes
        for _ in range(int(round(TICK_S / REPORT_S))):
            self.pending.extend(_reports(axes))

    def read(self, _size):
        return self.pending.pop(0) if self.pending else []


@unittest.skipUnless(LEGACY_ROBOSUITE, "LIBERO tests require .venv-libero (robosuite 1.4)")
class CollectionLoopSpaceMouseTests(unittest.TestCase):
    def run_layouts(self, reader_class) -> list:
        """Layout 0: hand resting on the cap, a push, R. Layout 1: cap untouched 1.5 s, then pushed."""
        import json
        import tempfile

        from wrist_teleop.libero_collect_teleop import run_libero_collect

        live = LiveDevice()

        def hook(env, info):
            if info.attempt == 1:
                live.set(BIAS if info.step < 10 else PUSH if 20 <= info.step < 30 else None)
                return "r" if info.step == 45 else None
            live.set(PUSH if 30 <= info.step < 50 else None)
            return "q" if info.step == 70 else None

        with tempfile.TemporaryDirectory() as folder:
            result = run_libero_collect(
                CONFIG, suite="libero_object", tasks=[1], demos_per_task=1, hand_mount="down", seed=7,
                headless=True, duration_s=40.0, camera_display=False, hand_control="off", view_size=(960, 540),
                planner_backend="straight", output_root=Path(folder) / "collect", log_root=Path(folder) / "logs",
                device=reader_class(CONFIG, hid_module=FakeHid(live)), step_hook=hook)
            self.assertEqual(result.exit_code, 0)
            lines = (result.run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            ended = [json.loads(line) for line in lines if '"attempt_end"' in line]
            exit_event = [json.loads(line) for line in lines if '"event": "exit"' in line or '"event":"exit"' in line]
            self.kept = exit_event[-1].get("spacemouse_kept_neutral_resets") if exit_event else None
            return [(e["outcome"], e["first_input_step"]) for e in ended]

    def test_the_spacemouse_drives_the_arm_right_after_a_new_layout(self):
        attempts = self.run_layouts(KeepNeutralSpaceMouse)
        self.assertEqual(attempts[0][0], "failed")  # moved, then R
        outcome, first = attempts[1]
        self.assertEqual(outcome, "quit")  # the push on layout 1 moved the arm at once
        self.assertLessEqual(first, 32)
        self.assertGreaterEqual(self.kept, 1)

    def test_the_old_reader_ignored_the_same_push(self):
        attempts = self.run_layouts(SpaceMouseHid)
        self.assertEqual(attempts[0][0], "failed")
        self.assertEqual(attempts[1], ("not_started", None))  # a whole second of pushing did nothing


if __name__ == "__main__":
    unittest.main()
