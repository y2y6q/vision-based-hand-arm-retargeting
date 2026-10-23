"""Reset and shutdown primitives used by the single integrated entry point."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .hand_gestures import AllegroHandGestureController, HandGestureCommand, HandGestureConfig
from .integrated import PandaAllegroActionComposer
from .runtime_controls import install_viewer_key_callback


def reset_integrated_environment(
    env: Any,
    *,
    mapper: Any,
    device: Any,
    hand_input: Any,
    gestures: AllegroHandGestureController | None = None,
    gesture_config: HandGestureConfig | None = None,
    buttons: Sequence[bool] = (),
    keypress_callback: Callable[[int], None] | None = None,
    timestamp_s: float | None = None,
    settle_duration_s: float = 0.0,
) -> tuple[PandaAllegroActionComposer, AllegroHandGestureController, HandGestureCommand, np.ndarray]:
    """Perform one complete R-key reset without reopening HID or camera.

    The reset order intentionally clears input state after robosuite rebuilds
    its world. A safe-open hand command is stepped before normal teleoperation
    resumes. ``settle_duration_s`` uses ordinary MuJoCo steps to let a mounted
    hand reach that safe-open target before the OSC goal is latched again. It
    matches the Shadow entry's reset discipline; it never writes object poses,
    disables gravity, or changes controller settings. Any button currently
    held seeds edge detection and therefore cannot retrigger a gesture
    immediately after reset.
    """
    env.reset()
    if keypress_callback is not None:
        install_viewer_key_callback(env, keypress_callback)
    composer = PandaAllegroActionComposer(env)
    composer.initialize_goal()
    if gestures is None:
        gestures = AllegroHandGestureController(
            HandGestureConfig() if gesture_config is None else gesture_config,
            joint_names=composer.hand_joint_names,
            lower_limits=composer.hand_lower_limits,
            upper_limits=composer.hand_upper_limits,
        )
    mapper.reset()
    device.reset_input_state()
    hand_input.reset(timestamp_s)
    button_state = tuple(buttons)
    required_button_count = max(
        gestures.config.left_button_index,
        gestures.config.right_button_index,
    ) + 1
    if len(button_state) < required_button_count:
        button_state = (False,) * required_button_count
    hand_command = gestures.reset(buttons=button_state, timestamp_s=timestamp_s)
    if not np.isfinite(settle_duration_s) or settle_duration_s < 0.0:
        raise ValueError("settle_duration_s must be a finite nonnegative number")
    action = composer.compose(np.zeros(6, dtype=np.float64), hand_command.joint_targets)
    if settle_duration_s > 0.0:
        control_hz = float(getattr(getattr(mapper, "config", None), "control_hz", 20.0))
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("mapper config must expose a positive control_hz")
        settle_steps = max(1, int(round(settle_duration_s * control_hz)))
        for _ in range(settle_steps):
            env.step(action)
        # The arm may have moved while the hand reached its bounded initial
        # target. Capture that settled pose so a neutral SpaceMouse begins at
        # an exact zero OSC increment, as the Shadow program does.
        composer.initialize_goal()
    env.step(action)
    return composer, gestures, hand_command, action


def safe_zero_step(env: Any, composer: PandaAllegroActionComposer) -> np.ndarray:
    """Send the final zero wrist delta while preserving the last safe hand pose."""
    action = composer.compose(np.zeros(6, dtype=np.float64), None)
    env.step(action)
    return action


def close_integrated_resources(
    *,
    env: Any | None,
    composer: PandaAllegroActionComposer | None,
    mapper: Any,
    device: Any,
    camera: Any,
) -> np.ndarray | None:
    """Best-effort ordered cleanup shared by Q, Ctrl+C, and normal exit."""
    last_action: np.ndarray | None = None
    errors: list[BaseException] = []
    try:
        if env is not None and composer is not None:
            last_action = safe_zero_step(env, composer)
    except BaseException as exc:
        errors.append(exc)
    try:
        mapper.reset()
    except BaseException as exc:
        errors.append(exc)
    try:
        device.close()
    except BaseException as exc:
        errors.append(exc)
    try:
        camera.close()
    except BaseException as exc:
        errors.append(exc)
    try:
        if env is not None:
            env.close()
    except BaseException as exc:
        errors.append(exc)
    if errors:
        first = errors[0]
        if len(errors) == 1:
            raise first
        messages = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        raise RuntimeError(f"multiple integrated cleanup failures: {messages}") from first
    return last_action
