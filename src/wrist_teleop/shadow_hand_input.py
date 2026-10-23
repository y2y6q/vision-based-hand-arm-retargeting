"""Camera landmark curls mapped safely to the official 20 Shadow actuators."""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence

import numpy as np

from .hand_gestures import (
    CAMERA_SOURCE,
    DEFAULT_HAND_GESTURE_CONFIG,
    FOUR_FINGER_PINCH,
    SAFE_OPEN_HOLD,
    THUMB_INDEX_PINCH,
    HandGestureCommand,
    HandGestureConfig,
)
from .hand_input import HandCommand, _point, estimate_all_finger_curls
from .shadow_gripper import SHADOW_ACTUATOR_NAMES


# [wrist_y, wrist_x, thumb x5, index x3, middle x3, ring x3, little x4]
# The wrist stays at its safe neutral target.  Values deliberately bias flexion
# joints and avoid forcing the lateral knuckles to their limits.
_CURL_WEIGHTS = np.array(
    [0.0, 0.0, 0.18, 0.62, 0.15, 0.74, 0.86, 0.16, 0.72, 0.84,
     0.16, 0.72, 0.84, 0.14, 0.66, 0.78, 0.14, 0.66, 0.78, 0.78],
    dtype=np.float64,
)

# Shadow's thumb is not a fourth ordinary finger.  THJ5 / THJ3 orient the
# thumb across the palm while THJ4 / THJ2 / THJ1 bend it.  A single positive
# "thumb curl" cannot represent that opposition motion, particularly when a
# person pinches with an extended thumb.  These are signed fractions of the
# available travel from the neutral (zero) target, in THJ5..THJ1 order.
#
# The index profile preserves the existing full-curl pose.  The middle profile
# was selected with the loaded right-hand MuJoCo model: it turns the thumb
# across the palm and reverses THJ2, reducing the thumb--middle tip-centre
# separation compared with the old all-positive curl mapping.
_THUMB_INDEX_PINCH_FRACTIONS = np.array([0.18, 0.62, 0.15, 0.74, 0.86], dtype=np.float64)
_THUMB_MIDDLE_PINCH_FRACTIONS = np.array([0.573, 0.90, 0.86, -0.358, 0.446], dtype=np.float64)
_THUMB_SLICE = slice(2, 7)
_PINCH_PARTNERS = ("index", "middle", "ring", "little")
_PINCH_CONTACT_RATIO = 0.35
_PINCH_RELEASE_RATIO = 0.90
_PINCH_PARTNER_CURL_FLOOR = 0.80
# Ignore weak incidental tip proximity. A real pinch normally yields a value
# close to one; the threshold prevents a nearly-open neighbor from changing
# the thumb's opposition profile frame to frame.
_PINCH_ACTIVATION_STRENGTH = 0.25


def estimate_thumb_pinch_strengths(landmarks: Sequence[Any]) -> np.ndarray:
    """Estimate thumb-tip contact with index, middle, ring, and little tips.

    The values are scale-normalized by the MCP span, so a hand moving closer
    to the webcam does not itself close the robot.  This deliberately adds a
    pinch cue to the existing curl estimator; it does not replace the normal
    per-finger curls used for free hand motion.
    """
    if len(landmarks) != 21:
        raise ValueError("MediaPipe hand input must contain exactly 21 landmarks")
    points = [_point(landmark) for landmark in landmarks]
    palm_span = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(palm_span) or palm_span < 1e-6:
        raise ValueError("MediaPipe palm span must be finite and non-zero")
    thumb_tip = points[4]
    ratios = np.array(
        [float(np.linalg.norm(thumb_tip - points[index])) / palm_span for index in (8, 12, 16, 20)],
        dtype=np.float64,
    )
    return np.clip(
        (_PINCH_RELEASE_RATIO - ratios) / (_PINCH_RELEASE_RATIO - _PINCH_CONTACT_RATIO),
        0.0,
        1.0,
    )


def _thumb_target_from_signed_fractions(
    fractions: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Convert safe signed neutral-relative thumb fractions into targets."""
    opened = np.clip(np.zeros(5, dtype=np.float64), lower, upper)
    positive_travel = upper - opened
    negative_travel = opened - lower
    travel = np.where(fractions >= 0.0, positive_travel, negative_travel)
    return np.clip(opened + fractions * travel, lower, upper)


def curls_to_shadow_targets(
    curls: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    *,
    thumb_pinch_strengths: Sequence[float] | None = None,
) -> np.ndarray:
    """Map MediaPipe's index/middle/ring/thumb curls to 20 actuator targets.

    The ring curl intentionally drives both ring and little fingers because
    the existing camera estimator supplies four curls, not five.  Tendon
    targets remain actuator values, not duplicated physical J2/J1 commands.
    """
    curls = np.asarray(curls, dtype=np.float64)
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    if curls.shape != (4,) or lower.shape != (20,) or upper.shape != (20,):
        raise ValueError("expected four curls and 20 ordered Shadow actuator limits")
    if not np.all(np.isfinite(curls)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(upper > lower):
        raise ValueError("curls and Shadow limits must be finite and ordered")
    resolved_curls = np.clip(curls, 0.0, 1.0).copy()
    strengths: np.ndarray | None = None
    active_partners = np.zeros(4, dtype=bool)
    pinch_strength = 0.0
    if thumb_pinch_strengths is not None:
        strengths = np.asarray(thumb_pinch_strengths, dtype=np.float64)
        if strengths.shape != (4,) or not np.all(np.isfinite(strengths)):
            raise ValueError("thumb_pinch_strengths must contain four finite values")
        strengths = np.clip(strengths, 0.0, 1.0)
        active_partners = strengths >= _PINCH_ACTIVATION_STRENGTH
        if np.any(active_partners):
            pinch_strength = float(np.max(strengths[active_partners]))
            # A human three-finger pinch exposes two thumb-tip contacts at
            # once. The original implementation kept only argmax(strengths),
            # so the other robot finger stayed open. Engage every confirmed
            # partner; ring and little intentionally share the legacy third
            # curl channel.
            for partner_index in np.flatnonzero(active_partners):
                curl_index = min(int(partner_index), 2)
                resolved_curls[curl_index] = max(
                    resolved_curls[curl_index],
                    float(strengths[partner_index]) * _PINCH_PARTNER_CURL_FLOOR,
                )

    # Existing estimator order is index, middle, ring/pinky, thumb.
    per_actuator_curl = np.array([
        0, 0, resolved_curls[3], resolved_curls[3], resolved_curls[3], resolved_curls[3], resolved_curls[3],
        resolved_curls[0], resolved_curls[0], resolved_curls[0],
        resolved_curls[1], resolved_curls[1], resolved_curls[1],
        resolved_curls[2], resolved_curls[2], resolved_curls[2], resolved_curls[2], resolved_curls[2], resolved_curls[2], resolved_curls[2],
    ], dtype=np.float64)
    opened = np.clip(np.zeros(20, dtype=np.float64), lower, upper)
    targets = opened + np.clip(per_actuator_curl, 0.0, 1.0) * _CURL_WEIGHTS * (upper - opened)
    targets = np.clip(targets, lower, upper)
    if strengths is None or pinch_strength <= 0.0:
        return targets

    # A thumb--index pinch remains the most compact opposition profile even
    # when middle also joins it for a tripod grip. Without index, a cross-palm
    # thumb profile brings the thumb toward middle / ring / little. This is a
    # deliberate contact-set choice, not an unstable argmax-only switch.
    profile = (
        _THUMB_INDEX_PINCH_FRACTIONS
        if active_partners[0]
        else _THUMB_MIDDLE_PINCH_FRACTIONS
    )
    opposition = _thumb_target_from_signed_fractions(profile, lower[_THUMB_SLICE], upper[_THUMB_SLICE])
    targets[_THUMB_SLICE] = (
        (1.0 - pinch_strength) * targets[_THUMB_SLICE] + pinch_strength * opposition
    )
    return np.clip(targets, lower, upper)


class ShadowLegacyHandInput:
    """Thread-safe latest-value source matching the existing camera worker API."""

    def __init__(self, lower_limits: Sequence[float], upper_limits: Sequence[float], smoothing_alpha: float = 0.45, stale_timeout_s: float = 0.5):
        self.lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(20)
        self.upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(20)
        if not np.all(np.isfinite(self.lower_limits)) or not np.all(np.isfinite(self.upper_limits)) or not np.all(self.upper_limits > self.lower_limits):
            raise ValueError("Shadow limits must be finite and ordered")
        if not 0.0 < float(smoothing_alpha) <= 1.0 or stale_timeout_s <= 0:
            raise ValueError("smoothing_alpha must be in (0, 1] and stale_timeout_s must be positive")
        self.joint_names = SHADOW_ACTUATOR_NAMES
        self.smoothing_alpha = float(smoothing_alpha)
        self.stale_timeout_s = float(stale_timeout_s)
        self._targets = np.clip(np.zeros(20), self.lower_limits, self.upper_limits)
        self._curls = np.zeros(4, dtype=np.float64)
        self._pinch_strengths = np.zeros(4, dtype=np.float64)
        self._timestamp = 0.0
        self._tracking_state = "waiting_for_hand"
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def calibration_state(self) -> str:
        return "not_required_rule_based_fingers"

    def _command_locked(self, valid: bool, timestamp: float | None = None) -> HandCommand:
        return HandCommand(
            joint_targets=self._targets.copy(), valid=valid,
            timestamp_s=self._timestamp if timestamp is None else timestamp,
            tracking_state=self._tracking_state, calibration_state=self.calibration_state,
            curls=self._curls.copy(), error=self._error,
            details={
                "algorithm": "shadow-legacy-curl-plus-pinch",
                "actuator_order": list(SHADOW_ACTUATOR_NAMES),
                "thumb_pinch_strengths": self._pinch_strengths.copy(),
                "thumb_pinch_partners": [
                    _PINCH_PARTNERS[index]
                    for index, strength in enumerate(self._pinch_strengths)
                    if strength >= _PINCH_ACTIVATION_STRENGTH
                ],
                "thumb_pinch_profile": (
                    "index"
                    if self._pinch_strengths[0] >= _PINCH_ACTIVATION_STRENGTH
                    else (
                        "cross_palm"
                        if float(np.max(self._pinch_strengths)) >= _PINCH_ACTIVATION_STRENGTH
                        else None
                    )
                ),
            },
        )

    def update_landmarks(self, landmarks: Sequence[Any], timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        try:
            curls = estimate_all_finger_curls(landmarks)
            pinch_strengths = estimate_thumb_pinch_strengths(landmarks)
            raw = curls_to_shadow_targets(
                curls, self.lower_limits, self.upper_limits,
                thumb_pinch_strengths=pinch_strengths,
            )
        except (TypeError, ValueError, IndexError) as exc:
            return self.mark_lost(timestamp, f"invalid landmarks: {exc}")
        with self._lock:
            self._targets = self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * self._targets
            self._targets = np.clip(self._targets, self.lower_limits, self.upper_limits)
            self._curls, self._pinch_strengths = curls, pinch_strengths
            self._timestamp, self._tracking_state, self._error = timestamp, "tracking", None
            return self._command_locked(valid=True)

    def mark_lost(self, timestamp_s: float | None = None, error: str | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            self._tracking_state, self._error = "lost_hold", error
            return self._command_locked(valid=False, timestamp=timestamp)

    def latest(self, timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            valid = self._tracking_state == "tracking" and timestamp - self._timestamp <= self.stale_timeout_s
            if not valid and self._tracking_state == "tracking":
                self._tracking_state = "stale_hold"
            return self._command_locked(valid=valid, timestamp=timestamp)

    def reset(self, timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            self._targets = np.clip(np.zeros(20), self.lower_limits, self.upper_limits)
            self._curls = np.zeros(4, dtype=np.float64)
            self._pinch_strengths = np.zeros(4, dtype=np.float64)
            self._timestamp, self._tracking_state, self._error = timestamp, "waiting_for_hand", None
            return self._command_locked(valid=False, timestamp=timestamp)


class ShadowHandGestureController:
    """Resolve camera targets and SpaceMouse button overrides for Shadow.

    This mirrors the Allegro gesture contract but produces targets in the
    official 20-actuator Shadow order.  It owns only hand targets: callers
    must pass the returned vector to :class:`PandaShadowActionComposer`, which
    keeps the Panda OSC slice independent.  Button 0 toggles thumb-index
    pinch; button 1 toggles four-finger pinch.  A held button cannot toggle a
    mode repeatedly because only rising edges are acted on.
    """

    def __init__(
        self,
        config: HandGestureConfig = DEFAULT_HAND_GESTURE_CONFIG,
        *,
        joint_names: Sequence[str] | None = None,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
    ):
        self.config = config
        self.joint_names = tuple(SHADOW_ACTUATOR_NAMES if joint_names is None else joint_names)
        if self.joint_names != SHADOW_ACTUATOR_NAMES:
            raise ValueError("gesture joint_names must match the official Shadow actuator order")
        self.lower_limits = self._limits("lower_limits", lower_limits)
        self.upper_limits = self._limits("upper_limits", upper_limits)
        if not np.all(self.upper_limits > self.lower_limits):
            raise ValueError("Shadow gesture joint limits must be ordered")

        self._targets = {
            # Use the same tested opposition geometry as camera pinches.
            THUMB_INDEX_PINCH: self._target_for_curls(
                config.thumb_index_pinch_curls, pinch_strengths=(1.0, 0.0, 0.0, 0.0)
            ),
            FOUR_FINGER_PINCH: self._target_for_curls(
                config.four_finger_pinch_curls, pinch_strengths=(1.0, 1.0, 1.0, 0.0)
            ),
            SAFE_OPEN_HOLD: self._target_for_curls(config.safe_open_curls),
        }
        self._previous_buttons: tuple[bool, ...] = ()
        self._active_override: str | None = None
        self._source = SAFE_OPEN_HOLD
        self._current_target = self._targets[SAFE_OPEN_HOLD].copy()
        self._transition_from = self._current_target.copy()
        self._transition_target = self._current_target.copy()
        self._transition_started_s: float | None = None

    @staticmethod
    def _limits(name: str, candidate: Sequence[float]) -> np.ndarray:
        values = np.asarray(candidate, dtype=np.float64)
        if values.shape != (20,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain 20 finite values")
        return values.copy()

    def _target_for_curls(
        self,
        curls: Sequence[float],
        *,
        pinch_strengths: Sequence[float] | None = None,
    ) -> np.ndarray:
        target = np.asarray(
            curls_to_shadow_targets(
                curls,
                self.lower_limits,
                self.upper_limits,
                thumb_pinch_strengths=pinch_strengths,
            ),
            dtype=np.float64,
        )
        if target.shape != (20,) or not np.all(np.isfinite(target)):
            raise RuntimeError("Shadow gesture mapping produced an invalid target")
        return np.clip(target, self.lower_limits, self.upper_limits)

    @property
    def active_override(self) -> str | None:
        """Held gesture name, or ``None`` while camera / safe-open controls hand."""
        return self._active_override

    @property
    def gesture_targets(self) -> dict[str, np.ndarray]:
        """Detached 20-actuator targets for diagnostics and regression tests."""
        return {name: target.copy() for name, target in self._targets.items()}

    def reset(
        self,
        buttons: Sequence[bool] | None = None,
        timestamp_s: float | None = None,
    ) -> HandGestureCommand:
        """Clear overrides and seed button state to avoid a reset-edge toggle."""
        timestamp = self._timestamp(timestamp_s)
        self._previous_buttons = self._normalize_buttons(buttons) if buttons is not None else ()
        self._active_override = None
        self._source = SAFE_OPEN_HOLD
        self._current_target = self._targets[SAFE_OPEN_HOLD].copy()
        self._transition_from = self._current_target.copy()
        self._transition_target = self._current_target.copy()
        self._transition_started_s = None
        return self._command(timestamp, (), "reset")

    def update(
        self,
        buttons: Sequence[bool],
        camera_targets: Sequence[float] | None,
        *,
        camera_valid: bool,
        timestamp_s: float | None = None,
    ) -> HandGestureCommand:
        """Return one bounded hand command from raw button and camera state."""
        timestamp = self._timestamp(timestamp_s)
        current_buttons = self._normalize_buttons(buttons)
        rising = self._rising_edges(current_buttons)
        self._previous_buttons = current_buttons
        event = self._apply_rising_edges(rising)

        desired_target, desired_source = self._select_target(camera_targets, camera_valid)
        self._advance_transition(timestamp)
        if desired_source != self._source:
            self._begin_transition(desired_target, desired_source, timestamp)
        elif desired_source in (CAMERA_SOURCE, SAFE_OPEN_HOLD):
            # A filtered camera target can change every tick.  Retarget a
            # running transition, otherwise adopt it directly.
            if self._transition_started_s is not None:
                self._transition_target = desired_target.copy()
            else:
                self._current_target = desired_target.copy()
        self._advance_transition(timestamp)
        return self._command(timestamp, rising, event)

    @staticmethod
    def _timestamp(value: float | None) -> float:
        timestamp = time.monotonic() if value is None else float(value)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        return timestamp

    def _normalize_buttons(self, values: Sequence[bool] | None) -> tuple[bool, ...]:
        if values is None:
            return ()
        if isinstance(values, (str, bytes)):
            raise ValueError("buttons must be a sequence of booleans")
        try:
            result = tuple(values)
        except TypeError as exc:
            raise ValueError("buttons must be a sequence of booleans") from exc
        required = max(self.config.left_button_index, self.config.right_button_index)
        if len(result) <= required:
            raise ValueError(f"buttons must contain index {required}")
        if any(not isinstance(value, (bool, np.bool_)) for value in result):
            raise ValueError("buttons must contain booleans")
        return tuple(bool(value) for value in result)

    def _rising_edges(self, current: tuple[bool, ...]) -> tuple[int, ...]:
        previous = self._previous_buttons
        return tuple(
            index
            for index in (self.config.left_button_index, self.config.right_button_index)
            if current[index] and (index >= len(previous) or not previous[index])
        )

    def _apply_rising_edges(self, rising: tuple[int, ...]) -> str | None:
        # If a malformed report raises both edges at once, process right last
        # so the four-finger gesture remains deterministic.
        event: str | None = None
        for index, gesture in (
            (self.config.left_button_index, THUMB_INDEX_PINCH),
            (self.config.right_button_index, FOUR_FINGER_PINCH),
        ):
            if index not in rising:
                continue
            if self._active_override == gesture:
                self._active_override = None
                event = f"deactivate_{gesture}"
            else:
                prior = self._active_override
                self._active_override = gesture
                event = f"activate_{gesture}" if prior is None else f"switch_{prior}_to_{gesture}"
        return event

    def _select_target(
        self,
        camera_targets: Sequence[float] | None,
        camera_valid: bool,
    ) -> tuple[np.ndarray, str]:
        if self._active_override is not None:
            return self._targets[self._active_override].copy(), self._active_override
        if camera_valid:
            target = self._camera_target(camera_targets)
            if target is not None:
                return target, CAMERA_SOURCE
        return self._targets[SAFE_OPEN_HOLD].copy(), SAFE_OPEN_HOLD

    def _camera_target(self, targets: Sequence[float] | None) -> np.ndarray | None:
        if targets is None:
            return None
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (20,) or not np.all(np.isfinite(values)):
            return None
        return np.clip(values, self.lower_limits, self.upper_limits)

    def _begin_transition(self, target: np.ndarray, source: str, timestamp: float) -> None:
        self._source = source
        self._transition_from = self._current_target.copy()
        self._transition_target = target.copy()
        if self.config.interpolation_duration_s == 0.0 or np.allclose(
            self._transition_from, self._transition_target, atol=1e-12, rtol=0.0
        ):
            self._current_target = target.copy()
            self._transition_started_s = None
        else:
            self._transition_started_s = timestamp

    def _advance_transition(self, timestamp: float) -> None:
        if self._transition_started_s is None:
            return
        duration = self.config.interpolation_duration_s
        progress = float(np.clip((timestamp - self._transition_started_s) / duration, 0.0, 1.0))
        self._current_target = self._transition_from + progress * (
            self._transition_target - self._transition_from
        )
        self._current_target = np.clip(self._current_target, self.lower_limits, self.upper_limits)
        if progress >= 1.0:
            self._current_target = self._transition_target.copy()
            self._transition_started_s = None

    def _command(
        self,
        timestamp: float,
        rising: tuple[int, ...],
        event: str | None,
    ) -> HandGestureCommand:
        if self._transition_started_s is None:
            transition_active = False
            transition_progress = 1.0
        else:
            transition_active = True
            transition_progress = float(np.clip(
                (timestamp - self._transition_started_s) / self.config.interpolation_duration_s,
                0.0,
                1.0,
            ))
        return HandGestureCommand(
            joint_targets=np.clip(self._current_target, self.lower_limits, self.upper_limits).copy(),
            source=self._source,
            active_override=self._active_override,
            transition_active=transition_active,
            transition_progress=transition_progress,
            button_rising_edges=rising,
            event=event,
            timestamp_s=timestamp,
        )
