"""Button-selectable, bounded Allegro hand gesture targets.

This module owns only the hand side of integrated teleoperation.  It receives
raw SpaceMouse button booleans and a camera-produced 16-joint target, then
returns one safe 16-joint target.  It never accepts or creates a Panda wrist
command.  The two button mappings are configured here rather than scattered
through the teleoperation loop:

* left button: ``thumb_index_pinch``
* right button: ``four_finger_pinch``

Gesture definitions are four curl fractions in the existing rule-based camera
mapping order: ``(index, middle, ring, thumb)``.  They are converted with the
same source-order joint limits as the live Allegro model, so every returned
target is finite and inside its corresponding physical limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
import time
from typing import Mapping, Sequence

import numpy as np

from .allegro_model import ALLEGRO_JOINT_NAMES, source_joint_specs
from .hand_input import curls_to_allegro_targets


THUMB_INDEX_PINCH = "thumb_index_pinch"
FOUR_FINGER_PINCH = "four_finger_pinch"
SAFE_OPEN_HOLD = "safe_open/hold"
CAMERA_SOURCE = "camera"
_OVERRIDE_SOURCES = frozenset((THUMB_INDEX_PINCH, FOUR_FINGER_PINCH))


def _validated_curls(name: str, values: Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain four finite curl fractions")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain four finite curl fractions") from exc
    if len(result) != 4:
        raise ValueError(f"{name} must contain four finite curl fractions")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in result):
        raise ValueError(f"{name} must contain four finite curl fractions")
    curls = tuple(float(value) for value in result)
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in curls):
        raise ValueError(f"{name} curl fractions must be finite values in [0, 1]")
    return curls  # type: ignore[return-value]


@dataclass(frozen=True)
class HandGestureConfig:
    """Centralized settings for SpaceMouse button hand gestures.

    ``*_curls`` use the legacy camera mapping order ``index, middle, ring,
    thumb``.  The defaults intentionally leave the middle and ring fingers
    open during the thumb-index pinch.  Button indices 0 and 1 are supported
    by recorded C62E raw-HID diagnostic reports; callers can override them
    for a different device report layout.
    """

    left_button_index: int = 0
    right_button_index: int = 1
    interpolation_duration_s: float = 0.20
    thumb_index_pinch_curls: tuple[float, float, float, float] = (0.85, 0.0, 0.0, 0.85)
    four_finger_pinch_curls: tuple[float, float, float, float] = (0.85, 0.80, 0.80, 0.85)
    safe_open_curls: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("left_button_index", "right_button_index"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= 31:
                raise ValueError(f"{name} must be an integer in [0, 31]")
            object.__setattr__(self, name, int(value))
        if self.left_button_index == self.right_button_index:
            raise ValueError("left_button_index and right_button_index must differ")
        duration = self.interpolation_duration_s
        if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real) or not np.isfinite(duration) or duration < 0.0:
            raise ValueError("interpolation_duration_s must be a finite value >= 0")
        object.__setattr__(self, "interpolation_duration_s", float(duration))
        for name in ("thumb_index_pinch_curls", "four_finger_pinch_curls", "safe_open_curls"):
            object.__setattr__(self, name, _validated_curls(name, getattr(self, name)))

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "HandGestureConfig":
        """Construct a validated configuration from a JSON-compatible mapping."""
        if not isinstance(values, Mapping):
            raise ValueError("hand gesture configuration must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown hand gesture settings: {sorted(unknown, key=str)}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-compatible settings for metadata logs."""
        return asdict(self)


DEFAULT_HAND_GESTURE_CONFIG = HandGestureConfig()


@dataclass(frozen=True)
class HandGestureCommand:
    """The resolved, absolute Allegro target for one integrated control tick."""

    joint_targets: np.ndarray
    source: str
    active_override: str | None
    transition_active: bool
    transition_progress: float
    button_rising_edges: tuple[int, ...]
    event: str | None
    timestamp_s: float


class AllegroHandGestureController:
    """Resolve camera targets and rising-edge button overrides safely.

    ``update`` has no dependency on robosuite and returns radians in the exact
    ``ALLEGRO_JOINT_NAMES`` source order.  It is deliberately stateful only
    for button edges and short target transitions.  The caller sends its
    output into :meth:`PandaAllegroActionComposer.compose`; that composer owns
    the hand action slice and keeps the Panda arm slice separate.
    """

    def __init__(
        self,
        config: HandGestureConfig = DEFAULT_HAND_GESTURE_CONFIG,
        *,
        joint_names: Sequence[str] | None = None,
        lower_limits: Sequence[float] | None = None,
        upper_limits: Sequence[float] | None = None,
    ):
        self.config = config
        specs = source_joint_specs()
        source_names = tuple(spec.name for spec in specs)
        source_lower = np.asarray([spec.lower for spec in specs], dtype=np.float64)
        source_upper = np.asarray([spec.upper for spec in specs], dtype=np.float64)
        if source_names != ALLEGRO_JOINT_NAMES:
            raise RuntimeError("Allegro source joint order changed unexpectedly")

        self.joint_names = tuple(ALLEGRO_JOINT_NAMES if joint_names is None else joint_names)
        if self.joint_names != source_names:
            raise ValueError("gesture joint_names must match the live Allegro source order")
        self.lower_limits = self._limits("lower_limits", lower_limits, source_lower)
        self.upper_limits = self._limits("upper_limits", upper_limits, source_upper)
        if not np.all(self.upper_limits > self.lower_limits):
            raise ValueError("Allegro gesture joint limits must be ordered")

        # Configured curl definitions become 16-vector radian targets through
        # the same mapping used by the camera adapter.
        self._targets = {
            THUMB_INDEX_PINCH: self._target_for_curls(config.thumb_index_pinch_curls),
            FOUR_FINGER_PINCH: self._target_for_curls(config.four_finger_pinch_curls),
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
    def _limits(name: str, candidate: Sequence[float] | None, fallback: np.ndarray) -> np.ndarray:
        values = fallback if candidate is None else np.asarray(candidate, dtype=np.float64)
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain 16 finite values")
        return values.copy()

    def _target_for_curls(self, curls: Sequence[float]) -> np.ndarray:
        target = np.asarray(curls_to_allegro_targets(curls, self.lower_limits, self.upper_limits), dtype=np.float64)
        if target.shape != (16,) or not np.all(np.isfinite(target)):
            raise RuntimeError("Allegro gesture mapping produced an invalid target")
        return np.clip(target, self.lower_limits, self.upper_limits)

    @property
    def active_override(self) -> str | None:
        """The held gesture, or ``None`` while camera / safe-open controls hand."""
        return self._active_override

    @property
    def gesture_targets(self) -> dict[str, np.ndarray]:
        """Copy the configured 16-joint gesture targets for logs and tests."""
        return {name: values.copy() for name, values in self._targets.items()}

    def reset(self, buttons: Sequence[bool] | None = None, timestamp_s: float | None = None) -> HandGestureCommand:
        """Clear override and interpolation, commanding the configured safe-open pose.

        Pass the currently observed button state during an environment reset.
        That seeds edge detection and prevents a button held across ``R`` from
        immediately reactivating its gesture in the next control tick.
        """
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
        """Resolve one tick of camera input and raw button state.

        Invalid or stale camera input does not reuse an unbounded historical
        camera target: when no button override is active the command moves to
        ``safe_open/hold``.  An active explicit gesture remains active until
        its configured button is clicked again or :meth:`reset` is called.
        """
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
            # Camera targets may legitimately change every frame.  Retarget a
            # source-change transition without restarting it each frame;
            # otherwise take the camera adapter's already filtered output.
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
        # A normal C62E report has one physical button transition at a time.
        # If both are reported in one packet, process the right button last so
        # four-finger pinch becomes the deterministic final mode.
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

    def _select_target(self, camera_targets: Sequence[float] | None, camera_valid: bool) -> tuple[np.ndarray, str]:
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
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            return None
        return np.clip(values, self.lower_limits, self.upper_limits)

    def _begin_transition(self, target: np.ndarray, source: str, timestamp: float) -> None:
        self._source = source
        self._transition_from = self._current_target.copy()
        self._transition_target = target.copy()
        if self.config.interpolation_duration_s == 0.0 or np.allclose(self._transition_from, self._transition_target, atol=1e-12, rtol=0.0):
            self._current_target = target.copy()
            self._transition_started_s = None
        else:
            self._transition_started_s = timestamp

    def _advance_transition(self, timestamp: float) -> None:
        if self._transition_started_s is None:
            return
        duration = self.config.interpolation_duration_s
        # Monotonic time should not go backward, but treating a synthetic
        # test's earlier timestamp as the start is safer than extrapolating.
        progress = float(np.clip((timestamp - self._transition_started_s) / duration, 0.0, 1.0))
        self._current_target = self._transition_from + progress * (self._transition_target - self._transition_from)
        self._current_target = np.clip(self._current_target, self.lower_limits, self.upper_limits)
        if progress >= 1.0:
            self._current_target = self._transition_target.copy()
            self._transition_started_s = None

    def _command(self, timestamp: float, rising: tuple[int, ...], event: str | None) -> HandGestureCommand:
        if self._transition_started_s is None:
            active = False
            progress = 1.0
        else:
            active = True
            progress = float(np.clip(
                (timestamp - self._transition_started_s) / self.config.interpolation_duration_s,
                0.0,
                1.0,
            ))
        return HandGestureCommand(
            joint_targets=np.clip(self._current_target, self.lower_limits, self.upper_limits).copy(),
            source=self._source,
            active_override=self._active_override,
            transition_active=active,
            transition_progress=progress,
            button_rising_edges=rising,
            event=event,
            timestamp_s=timestamp,
        )
