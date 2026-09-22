"""Thread-safe R/Q routing and a local MuJoCo viewer callback wrapper.

robosuite 1.5.2's ``MjviewerRenderer.add_keypress_callback`` stores a
callback but does not pass it to MuJoCo's ``launch_passive``.  This module
keeps the workaround in project code: it replaces only the per-environment
renderer wrapper and never edits installed packages.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class KeyboardCommandRouter:
    """Convert R/Q callbacks into one-shot main-loop requests safely.

    Native MuJoCo key callbacks carry only a key code, not a release event.
    A quiet interval therefore acts as release detection: auto-repeat updates
    the interval and cannot repeatedly reset an environment, while a later
    fresh press is accepted after the key has been quiet long enough.
    """

    def __init__(self, repeat_release_s: float = 0.25, clock: Callable[[], float] = time.monotonic):
        if repeat_release_s <= 0:
            raise ValueError("repeat_release_s must be positive")
        self.repeat_release_s = float(repeat_release_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self._reset_pending = False
        self._stop_event = threading.Event()
        self._quit_reason: str | None = None

    @staticmethod
    def _normalize_key(key: object) -> str | None:
        if isinstance(key, str):
            return key[:1].lower() if key else None
        if isinstance(key, int) and 0 <= key <= 255:
            return chr(key).lower()
        return None

    def feed(self, key: object, *, source: str = "keyboard", timestamp_s: float | None = None) -> str | None:
        """Queue ``reset`` or ``quit`` once; harmless keys return ``None``."""
        normalized = self._normalize_key(key)
        if normalized not in {"r", "q"}:
            return None
        timestamp = self._clock() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            previous = self._last_seen.get(normalized)
            self._last_seen[normalized] = timestamp
            if previous is not None and timestamp - previous < self.repeat_release_s:
                return None
            if normalized == "r":
                if self._stop_event.is_set():
                    return None
                self._reset_pending = True
                return "reset"
            self._stop_event.set()
            self._quit_reason = f"{source}:Q"
            return "quit"

    def take_reset(self) -> bool:
        """Consume exactly one queued reset request from the control loop."""
        with self._lock:
            requested, self._reset_pending = self._reset_pending, False
            return requested

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def quit_reason(self) -> str | None:
        with self._lock:
            return self._quit_reason


class CallbackMjviewerRenderer:
    """Small local equivalent of robosuite's MjviewerRenderer with R/Q hook."""

    def __init__(self, env: Any, camera_id: int | None, camera_config: dict[str, Any] | None,
                 keypress_callback: Callable[[int], None]):
        self.env = env
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.keypress_callback = keypress_callback
        self.viewer = None

    def render(self) -> None:
        pass

    def set_camera(self, camera_id: int) -> None:
        self.camera_id = camera_id

    def update(self) -> None:
        if self.viewer is None:
            from mujoco import viewer as mujoco_viewer

            self.viewer = mujoco_viewer.launch_passive(
                self.env.sim.model._model,
                self.env.sim.data._data,
                key_callback=self.keypress_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.opt.geomgroup[0] = 0
            if self.camera_config is not None:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]
            if self.camera_id is not None:
                if self.camera_id >= 0:
                    self.viewer.cam.type = 2
                    self.viewer.cam.fixedcamid = self.camera_id
                else:
                    self.viewer.cam.type = 0
        self.viewer.sync()

    def reset(self) -> None:
        pass

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def add_keypress_callback(self, keypress_callback: Callable[[int], None]) -> None:
        self.keypress_callback = keypress_callback


def install_viewer_key_callback(env: Any, callback: Callable[[int], None]) -> bool:
    """Install project-local R/Q handling on this env's current viewer wrapper.

    Return ``False`` for a headless environment. Call this after every
    ``env.reset()`` because robosuite destroys and recreates its mjviewer
    wrapper on a hard reset.
    """
    renderer = getattr(env, "viewer", None)
    if renderer is None:
        return False
    if renderer.__class__.__name__ != "MjviewerRenderer":
        add_callback = getattr(renderer, "add_keypress_callback", None)
        if callable(add_callback):
            add_callback(callback)
            return True
        return False
    native = getattr(renderer, "viewer", None)
    if native is not None:
        renderer.close()
    env.viewer = CallbackMjviewerRenderer(
        env,
        camera_id=getattr(renderer, "camera_id", None),
        camera_config=getattr(renderer, "camera_config", None),
        keypress_callback=callback,
    )
    return True
