"""Windows HID acquisition for a 3Dconnexion SpaceMouse.

This module deliberately exposes raw motion and button state only.  It does
not know about grippers, Allegro hands, or robosuite actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np

from .config import WristConfig


SPACEMOUSE_VENDOR_ID = 0x256F


@dataclass(frozen=True)
class MotionSample:
    """One fail-closed SpaceMouse sample in HID report-axis order.

    ``raw_axes`` is ``[report_tx, report_ty, report_tz, roll, pitch, yaw]``.
    The mapper owns all axis reordering, signs, scale, and frame conversion.
    ``valid`` applies only to motion and becomes false after stale motion or a
    HID failure. ``buttons_valid`` tracks the independent button packet
    freshness, so a stationary wrist can still receive a button rising edge
    without replaying an old motion vector.
    """

    raw_axes: np.ndarray
    buttons: tuple[bool, ...]
    valid: bool
    armed: bool
    timestamp_s: float
    error: str | None = None
    buttons_valid: bool = False
    calibrated_axes: np.ndarray | None = None
    zero_bias_axes: np.ndarray | None = None
    neutral_locked: bool = False
    calibration_state: str = "waiting_for_neutral"

    @property
    def control_axes(self) -> np.ndarray:
        """Return startup-calibrated axes, or raw axes for legacy test samples.

        Keeping ``raw_axes`` untouched makes every telemetry record useful for
        diagnosing HID noise. Real ``SpaceMouseHid`` samples always provide a
        calibrated vector; synthetic / legacy callers safely retain the raw
        vector until they are updated.
        """
        return self.raw_axes if self.calibrated_axes is None else self.calibrated_axes


class SpaceMouseHid:
    """Non-blocking HID reader with explicit stale and disconnect handling.

    SpaceMouse Wireless units can report translation and rotation in separate
    packets (IDs 1 and 2), while some models use a single 13-byte ID-1 packet.
    Both layouts are decoded here.  A stale or failed read never replays the
    previous motion command.
    """

    def __init__(self, config: WristConfig, hid_module: Any | None = None):
        self.config = config
        if hid_module is None:
            try:
                import hid as hid_module  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:  # pragma: no cover - install-time failure
                raise RuntimeError("hidapi is required; install the locked robosuite environment") from exc
        self._hid = hid_module
        self._device: Any | None = None
        self._raw_axes = np.zeros(6, dtype=np.float64)
        self._buttons = (False,) * 8
        self._opened_at: float | None = None
        self._last_motion_at: float | None = None
        self._last_button_at: float | None = None
        self._armed = False
        self._error: str | None = None
        self._zero_bias = np.zeros(6, dtype=np.float64)
        self._zero_bias_samples: list[np.ndarray] = []
        self._zero_bias_locked = False
        self._calibration_state = "waiting_for_neutral"
        self.device_info: dict[str, Any] | None = None

    @staticmethod
    def _safe_text(value: Any) -> Any:
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else value

    def discover(self) -> list[dict[str, Any]]:
        """Return compatible 3Dconnexion HID interfaces without opening one."""
        vendor_id = SPACEMOUSE_VENDOR_ID if self.config.vendor_id is None else self.config.vendor_id
        product_id = self.config.product_id
        devices = []
        for item in self._hid.enumerate():
            if item.get("vendor_id") != vendor_id:
                continue
            if product_id is not None and item.get("product_id") != product_id:
                continue
            devices.append({key: self._safe_text(value) for key, value in item.items()})
        return devices

    def open(self) -> dict[str, Any]:
        """Open the first configured SpaceMouse interface and reset all state."""
        matches = self.discover()
        if not matches:
            raise OSError("No matching 3Dconnexion SpaceMouse HID interface was found")
        selected = matches[0]
        device = self._hid.device()
        path = selected["path"]
        # hidapi on this Windows build enumerates a str but requires bytes.
        device.open_path(path.encode() if isinstance(path, str) else path)
        device.set_nonblocking(1)
        self._device = device
        self.device_info = selected
        self.reset_input_state()
        return selected

    def reset_input_state(self) -> None:
        """Discard motion, buttons, and filter arming without closing HID.

        Environment reset uses this to prevent a pre-reset motion or button
        packet from producing an action after reset. The open handle remains
        intact, and normal neutral-startup arming applies again.
        """
        self._raw_axes.fill(0.0)
        self._buttons = (False,) * 8
        self._opened_at = time.monotonic() if self._device is not None else None
        self._last_motion_at = None
        self._last_button_at = None
        self._armed = False
        self._error = None
        self._zero_bias.fill(0.0)
        self._zero_bias_samples.clear()
        self._zero_bias_locked = False
        self._calibration_state = "waiting_for_neutral"

    def _observe_startup_neutral(self) -> None:
        """Lock a fixed raw zero bias after a short centred startup window.

        This deliberately runs only before arming. Once locked, no live input
        can alter the offset, so ordinary operator motion cannot be mistaken
        for a new neutral. Each decoded HID motion report contributes one
        sample even when multiple reports are drained in a single poll.
        """
        if self._zero_bias_locked:
            return
        thresholds = np.asarray(self.config.raw_ranges, dtype=np.float64) * np.array(
            [self.config.translation_deadzone] * 3 + [self.config.rotation_deadzone] * 3,
            dtype=np.float64,
        )
        if not np.all(np.abs(self._raw_axes) <= thresholds):
            self._zero_bias_samples.clear()
            self._calibration_state = "waiting_for_neutral"
            return
        self._zero_bias_samples.append(self._raw_axes.copy())
        if len(self._zero_bias_samples) < self.config.zero_bias_calibration_frames:
            self._calibration_state = "calibrating_neutral"
            return
        # Median rejects an occasional centred-report spike without adapting
        # to any later operator movement.
        self._zero_bias[:] = np.median(np.stack(self._zero_bias_samples, axis=0), axis=0)
        self._zero_bias_samples.clear()
        self._zero_bias_locked = True
        self._calibration_state = "neutral_locked"

    @staticmethod
    def _int16(low: int, high: int) -> int:
        value = int(low) | (int(high) << 8)
        return value - 65536 if value >= 32768 else value

    @classmethod
    def _three_axes(cls, report: Iterable[int]) -> np.ndarray:
        values = list(report)
        if len(values) < 7:
            raise ValueError("short six-axis HID report")
        # Keep HID report order. The centrally configured mapper implements
        # the robosuite-compatible x/y reorder and z inversion.
        return np.array([
            cls._int16(values[1], values[2]),
            cls._int16(values[3], values[4]),
            cls._int16(values[5], values[6]),
        ], dtype=np.float64)

    def _decode(self, report: Iterable[int], now: float) -> None:
        values = list(report)
        if not values:
            return
        report_id = int(values[0])
        if report_id == 1:
            self._raw_axes[:3] = self._three_axes(values)
            if len(values) >= 13:
                self._raw_axes[3:] = np.array([
                    self._int16(values[7], values[8]),
                    self._int16(values[9], values[10]),
                    self._int16(values[11], values[12]),
                ], dtype=np.float64)
            self._last_motion_at = now
            self._observe_startup_neutral()
        elif report_id == 2:
            self._raw_axes[3:] = self._three_axes(values)
            self._last_motion_at = now
            self._observe_startup_neutral()
        elif report_id == 3 and len(values) >= 2:
            mask = int(values[1])
            self._buttons = tuple(bool(mask & (1 << bit)) for bit in range(8))
            self._last_button_at = now

    def poll(self) -> MotionSample:
        """Drain pending reports and return the latest safe sample.

        A device that stops producing motion reports becomes invalid after
        ``stale_timeout_s``.  The consumer must map invalid samples to zero.
        """
        now = time.monotonic()
        if self._device is None:
            return MotionSample(np.zeros(6), self._buttons, False, False, now, self._error or "not connected", False)
        try:
            while True:
                report = self._device.read(64)
                if not report:
                    break
                self._decode(report, now)
        except Exception as exc:  # HID exceptions are hardware failures; fail closed.
            self._error = f"HID read failed: {type(exc).__name__}: {exc}"
            self.close()
            return MotionSample(np.zeros(6), self._buttons, False, False, now, self._error, False)

        calibrated = self._raw_axes - self._zero_bias if self._zero_bias_locked else np.zeros(6, dtype=np.float64)
        neutral_limit = np.asarray(self.config.raw_ranges) * np.array([
            self.config.translation_deadzone,
        ] * 3 + [self.config.rotation_deadzone] * 3)
        if (not self._armed and self._opened_at is not None
                and now - self._opened_at >= self.config.neutral_startup_s
                and self._zero_bias_locked
                and np.all(np.abs(calibrated) <= neutral_limit)):
            self._armed = True
        valid = self._last_motion_at is not None and now - self._last_motion_at <= self.config.stale_timeout_s
        buttons_valid = self._last_button_at is not None and now - self._last_button_at <= self.config.stale_timeout_s
        control_axes = calibrated.copy() if valid and self._zero_bias_locked else np.zeros(6, dtype=np.float64)
        return MotionSample(
            self._raw_axes.copy(), self._buttons, valid, self._armed, now, self._error, buttons_valid,
            control_axes, self._zero_bias.copy(), self._zero_bias_locked, self._calibration_state,
        )

    def close(self) -> None:
        """Release the HID handle and make subsequent samples safe zeros."""
        device, self._device = self._device, None
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        self._raw_axes.fill(0.0)
        self._buttons = (False,) * 8
        self._opened_at = None
        self._last_motion_at = None
        self._last_button_at = None
        self._armed = False
        self._zero_bias.fill(0.0)
        self._zero_bias_samples.clear()
        self._zero_bias_locked = False
        self._calibration_state = "closed"

    def __enter__(self) -> "SpaceMouseHid":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
