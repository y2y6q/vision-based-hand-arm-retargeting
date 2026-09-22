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
) -> tuple[PandaAllegroActionComposer, AllegroHandGestureController, HandGestureCommand, np.ndarray]:
    """Perform one complete R-key reset without reopening HID or camera.

    The reset order intentionally clears input state after robosuite rebuilds
    its world. A safe-open hand command is stepped once so the new Allegro
    position actuators receive a bounded target before normal teleoperation
    resumes. Any button currently held seeds edge detection and therefore
    cannot retrigger a gesture immediately after reset.
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
    action = composer.compose(np.zeros(6, dtype=np.float64), hand_command.joint_targets)
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
