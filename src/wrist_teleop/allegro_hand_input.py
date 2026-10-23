"""Shadow-style camera mapping for the Allegro hand: curls plus a pinch cue.

This follows the Panda + Shadow entry's rule-based mapper: per-finger curl
from MediaPipe landmark angles (independent of camera mirroring and of the
first frame seen), plus a thumb-tip-to-fingertip pinch cue. A detected pinch
closes the partner finger to at least its pinch curl and blends the thumb
into an opposition pose fitted on the loaded Allegro model.

The legacy Allegro thumb curl alone stops 30.5 mm from the curled index tip.
The opposition fractions below (joint 12..15, of the travel from the
safe-open target to the upper limit) were found by forward-kinematics search
on the 2/3-scale MuJoCo Allegro with the partner finger at 0.8 curl; each
brings the thumb tip to fingertip contact distance (16 mm centre-to-centre).

The thumb itself is driven by two continuous cues instead of one curl:
opposition (how far the thumb tip has crossed the palm, from its distance to
the ring-finger MCP) and flexion (the legacy thumb curl). On the Allegro
model joint 12 swings the thumb under the palm, joint 13 rolls its pad toward
the fingers, and joints 14 / 15 flex it, so opposition drives 12 / 13 and
flexion drives 14 / 15. This lets the thumb cross the palm for power and
tripod grasps without first needing a detected pinch.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .hand_input import (
    DEFAULT_ALLEGRO_JOINT_NAMES,
    DEFAULT_ALLEGRO_LOWER_LIMITS,
    DEFAULT_ALLEGRO_UPPER_LIMITS,
    HandCommand,
    HandInputAdapter,
    _point,
    curls_to_allegro_targets,
    estimate_all_finger_curls,
)
from .shadow_hand_input import (
    _PINCH_ACTIVATION_STRENGTH,
    _PINCH_PARTNER_CURL_FLOOR,
    _PINCH_PARTNERS,
    estimate_thumb_pinch_strengths,
)


_THUMB_SLICE = slice(12, 16)
_ALLEGRO_THUMB_OPPOSITION_FRACTIONS = {
    "index": np.array([0.333, 0.417, 0.500, 0.667], dtype=np.float64),
    "middle": np.array([0.583, 0.583, 0.667, 0.750], dtype=np.float64),
    "ring": np.array([0.750, 0.750, 0.583, 0.583], dtype=np.float64),
}
# Human index, middle, ring, little -> Allegro index, middle, ring curl slot.
_PARTNER_TO_CURL_SLOT = (0, 1, 2, 2)
# Thumb-tip to ring-MCP distance in palm widths: ~1.8 open, ~1.1-1.3 beside
# the index, ~0.5 across the palm.
_OPPOSITION_OPEN_RATIO = 1.3
_OPPOSITION_FULL_RATIO = 0.5
# Fractions of joint 12..15 travel from safe-open. Full opposition with half
# flexion lands next to the fitted middle-finger opposition pose.
_THUMB_OPPOSITION_WEIGHTS = np.array([0.60, 0.60, 0.35, 0.25], dtype=np.float64)
_THUMB_FLEXION_WEIGHTS = np.array([0.15, 0.00, 0.70, 0.80], dtype=np.float64)


def estimate_thumb_opposition(landmarks: Sequence[Any]) -> float:
    """0 = thumb out to the side, 1 = thumb tip across the palm (scale-free)."""

    if len(landmarks) != 21:
        raise ValueError("MediaPipe hand input must contain exactly 21 landmarks")
    points = [_point(landmark) for landmark in landmarks]
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(palm_width) or palm_width < 1e-6:
        raise ValueError("MediaPipe palm width must be finite and non-zero")
    ratio = float(np.linalg.norm(points[4] - points[13])) / palm_width
    return float(np.clip(
        (_OPPOSITION_OPEN_RATIO - ratio) / (_OPPOSITION_OPEN_RATIO - _OPPOSITION_FULL_RATIO), 0.0, 1.0
    ))


def thumb_targets(opposition: float, flexion: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Allegro joint 12..15 targets from the two thumb cues."""

    opened = np.clip(np.zeros(4), lower[_THUMB_SLICE], upper[_THUMB_SLICE])
    fractions = np.clip(
        float(np.clip(opposition, 0.0, 1.0)) * _THUMB_OPPOSITION_WEIGHTS
        + float(np.clip(flexion, 0.0, 1.0)) * _THUMB_FLEXION_WEIGHTS,
        0.0, 1.0,
    )
    return opened + fractions * (upper[_THUMB_SLICE] - opened)


def curls_to_allegro_pinch_targets(
    curls: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    *,
    thumb_pinch_strengths: Sequence[float] | None = None,
    thumb_opposition: float | None = None,
) -> tuple[np.ndarray, str | None]:
    """Return 16 Allegro targets and the active thumb profile (or ``None``).

    ``thumb_opposition=None`` keeps the legacy single-curl thumb.
    """

    resolved = np.clip(np.asarray(curls, dtype=np.float64), 0.0, 1.0).copy()
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    if thumb_pinch_strengths is None:
        targets = curls_to_allegro_targets(resolved, lower, upper)
        if thumb_opposition is not None:
            targets[_THUMB_SLICE] = thumb_targets(thumb_opposition, resolved[3], lower, upper)
        return targets, None
    strengths = np.clip(np.asarray(thumb_pinch_strengths, dtype=np.float64), 0.0, 1.0)
    if strengths.shape != (4,) or not np.all(np.isfinite(strengths)):
        raise ValueError("thumb_pinch_strengths must contain four finite values")
    active = strengths >= _PINCH_ACTIVATION_STRENGTH
    for partner in np.flatnonzero(active):
        slot = _PARTNER_TO_CURL_SLOT[int(partner)]
        resolved[slot] = max(resolved[slot], float(strengths[partner]) * _PINCH_PARTNER_CURL_FLOOR)
    targets = curls_to_allegro_targets(resolved, lower, upper)
    if thumb_opposition is not None:
        targets[_THUMB_SLICE] = thumb_targets(thumb_opposition, resolved[3], lower, upper)
    if not np.any(active):
        return targets, None
    # Prefer the most radial partner present, as the Shadow mapper does:
    # index pinch stays compact even when middle joins for a tripod grip.
    profile = "index" if active[0] else ("middle" if active[1] else "ring")
    strength = float(np.max(strengths[active]))
    opened = np.clip(np.zeros(4), lower[_THUMB_SLICE], upper[_THUMB_SLICE])
    opposition = opened + _ALLEGRO_THUMB_OPPOSITION_FRACTIONS[profile] * (upper[_THUMB_SLICE] - opened)
    targets[_THUMB_SLICE] = (1.0 - strength) * targets[_THUMB_SLICE] + strength * opposition
    return np.clip(targets, lower, upper), profile


class AllegroCurlPinchHandInput(HandInputAdapter):
    """Camera source with the Shadow entry's curl + pinch logic for Allegro."""

    def __init__(
        self,
        lower_limits: Sequence[float] = DEFAULT_ALLEGRO_LOWER_LIMITS,
        upper_limits: Sequence[float] = DEFAULT_ALLEGRO_UPPER_LIMITS,
        joint_names: Sequence[str] = DEFAULT_ALLEGRO_JOINT_NAMES,
        smoothing_alpha: float = 0.45,
        stale_timeout_s: float = 0.5,
    ):
        super().__init__(lower_limits, upper_limits, joint_names, smoothing_alpha, stale_timeout_s)
        self._pinch_strengths = np.zeros(4, dtype=np.float64)
        self._profile: str | None = None
        self._opposition = 0.0

    def update_landmarks(self, landmarks: Sequence[Any], timestamp_s: float | None = None) -> HandCommand:
        import time

        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        try:
            curls = estimate_all_finger_curls(landmarks)
            strengths = estimate_thumb_pinch_strengths(landmarks)
            opposition = estimate_thumb_opposition(landmarks)
            raw_targets, profile = curls_to_allegro_pinch_targets(
                curls, self.lower_limits, self.upper_limits,
                thumb_pinch_strengths=strengths, thumb_opposition=opposition,
            )
        except (TypeError, ValueError, IndexError) as exc:
            return self.mark_lost(timestamp, f"invalid landmarks: {exc}")
        with self._lock:
            self._targets = self.smoothing_alpha * raw_targets + (1.0 - self.smoothing_alpha) * self._targets
            self._targets = np.clip(self._targets, self.lower_limits, self.upper_limits)
            self._curls, self._pinch_strengths, self._profile = curls, strengths, profile
            self._opposition = opposition
            self._timestamp = timestamp
            self._tracking_state = "tracking"
            self._error = None
            return self._command_locked(valid=True)

    def reset(self, timestamp_s: float | None = None) -> HandCommand:
        with self._lock:
            self._pinch_strengths = np.zeros(4, dtype=np.float64)
            self._profile = None
            self._opposition = 0.0
        return super().reset(timestamp_s)

    def _command_locked(self, valid: bool, timestamp: float | None = None) -> HandCommand:
        command = super()._command_locked(valid, timestamp)
        command.details.update({
            "algorithm": "curl-pinch (Shadow-style rule mapping)",
            "thumb_pinch_strengths": self._pinch_strengths.copy(),
            "thumb_pinch_partners": [
                _PINCH_PARTNERS[index] for index, value in enumerate(self._pinch_strengths)
                if value >= _PINCH_ACTIVATION_STRENGTH
            ],
            "thumb_pinch_profile": self._profile,
            "thumb_opposition": self._opposition,
        })
        return command
