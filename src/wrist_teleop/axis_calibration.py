"""Solve SpaceMouse axis order / signs from six guided physical gestures.

The control basis is the primary camera's ``[screen right, away from the
operator, world up]``. For rotations the hand follows the cap: tilting the
top away from you is a negative rotation about "right", tilting the top to
the right is positive about "away", and a clockwise twist seen from above is
negative about "up" (right-hand rule in that basis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AxisGesture:
    key: str
    prompt: str
    control_index: int
    control_sign: int


AXIS_GESTURES: tuple[AxisGesture, ...] = (
    AxisGesture("push_away", "把帽子水平向前推（远离自己），保持住", 1, +1),
    AxisGesture("push_right", "把帽子水平向右推，保持住", 0, +1),
    AxisGesture("lift_up", "把帽子向上提起，保持住", 2, +1),
    AxisGesture("tilt_away", "把帽子顶部向前倾（像低头），保持住", 3, -1),
    AxisGesture("tilt_right", "把帽子顶部向右倾，保持住", 4, +1),
    AxisGesture("twist_clockwise", "从上往下看，顺时针拧帽子，保持住", 5, -1),
)


@dataclass(frozen=True)
class GestureMeasurement:
    gesture: AxisGesture
    raw_axis: int
    raw_sign: int
    purity: float
    peak: float


def measure_gesture(gesture: AxisGesture, samples: Sequence[Sequence[float]], *, min_peak: float = 60.0) -> GestureMeasurement:
    """Pick the dominant raw axis of the gesture's own group (translation / rotation)."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("need at least one 6-axis sample")
    group = slice(0, 3) if gesture.control_index < 3 else slice(3, 6)
    offset = group.start
    part = values[:, group]
    strength = np.abs(part).max(axis=1)
    held = part[strength >= 0.5 * strength.max()] if strength.max() > 0 else part
    mean = held.mean(axis=0)
    axis = int(np.argmax(np.abs(mean)))
    peak = float(abs(mean[axis]))
    if peak < min_peak:
        raise ValueError(f"{gesture.key}: motion too small (peak {peak:.0f}); push further and hold")
    purity = peak / float(np.linalg.norm(mean))
    return GestureMeasurement(gesture, offset + axis, 1 if mean[axis] > 0 else -1, purity, peak)


def solve_axis_mapping(measurements: Sequence[GestureMeasurement], *, min_purity: float = 0.75) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(axis_order, axis_inversion)`` for ``WristConfig``."""

    order = [-1] * 6
    inversion = [1] * 6
    for item in measurements:
        if item.purity < min_purity:
            raise ValueError(
                f"{item.gesture.key}: gesture mixed several axes (purity {item.purity:.2f}); repeat it more cleanly"
            )
        j = item.gesture.control_index
        order[j] = item.raw_axis
        inversion[j] = item.gesture.control_sign * item.raw_sign
    if sorted(order[:3]) != [0, 1, 2] or sorted(order[3:]) != [3, 4, 5]:
        raise ValueError(f"gestures did not identify distinct axes: axis_order={order}")
    return tuple(order), tuple(inversion)
