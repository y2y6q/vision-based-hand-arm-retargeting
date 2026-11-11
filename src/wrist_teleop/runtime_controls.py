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

# R presses this soon after a held reset ends are still dropped: they were
# pressed while the picture was frozen, or reach us late from a busy window.
RESET_RELEASE_GRACE_S = 0.5
# Drawn across the main view while a reset runs (OpenCV text: ASCII only).
RESET_NOTICE = "RESETTING - please wait\nR is ignored until the scene is back"
# Longest wait for the notice to reach the screen before the reset blocks.
RESET_NOTICE_WAIT_S = 0.15


class KeyboardCommandRouter:
    """Convert R/Q callbacks into one-shot main-loop requests safely.

    Native MuJoCo key callbacks carry only a key code, not a release event.
    A quiet interval therefore acts as release detection: auto-repeat updates
    the interval and cannot repeatedly reset an environment, while a later
    fresh press is accepted after the key has been quiet long enough.

    ``hold_resets`` / ``release_resets`` drop R while a slow reset runs
    (the windows keep reading keys while the control loop is blocked); Q is
    never held.
    """

    def __init__(self, repeat_release_s: float = 0.25, clock: Callable[[], float] = time.monotonic):
        if repeat_release_s <= 0:
            raise ValueError("repeat_release_s must be positive")
        self.repeat_release_s = float(repeat_release_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self._reset_pending = False
        self._reset_held = False
        self._ignore_resets_until = float("-inf")
        self._ignored_resets = 0
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
                if self._reset_held or timestamp < self._ignore_resets_until:
                    self._ignored_resets += 1
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

    def hold_resets(self) -> None:
        """Drop every R from now until ``release_resets`` (call as a slow reset starts)."""
        with self._lock:
            self._reset_held = True
            self._reset_pending = False
            self._ignored_resets = 0

    def release_resets(self, grace_s: float = RESET_RELEASE_GRACE_S) -> int:
        """Accept R again ``grace_s`` from now; returns how many R were dropped while held."""
        with self._lock:
            self._reset_held = False
            self._reset_pending = False
            self._ignore_resets_until = self._clock() + max(0.0, float(grace_s))
            ignored, self._ignored_resets = self._ignored_resets, 0
            return ignored

    @property
    def resets_held(self) -> bool:
        with self._lock:
            return self._reset_held

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def quit_reason(self) -> str | None:
        with self._lock:
            return self._quit_reason


class ResetFeedback:
    """One R = one reset, and the reset is visible while it runs.

    A LIBERO hard reset blocks the control loop for ~1.9 s and the first step
    on the new model for another ~0.5 s while the last picture stays on
    screen. The windows keep reading keys meanwhile, so an R pressed because
    "nothing happens" used to queue one more reset (2026-11-07: four in a
    row, 7.6 s frozen). ``begin`` (when the loop takes a reset) holds R and
    puts a notice on the view; ``settle`` (after the first step on the new
    model) removes it and accepts R again after ``grace_s``.
    """

    def __init__(self, keyboard: KeyboardCommandRouter, *, grace_s: float = RESET_RELEASE_GRACE_S) -> None:
        self.keyboard = keyboard
        self.grace_s = float(grace_s)
        self._started: float | None = None
        self._label = "reset"

    @property
    def active(self) -> bool:
        return self._started is not None

    def begin(self, view: Any = None, *, label: str = "reset", notice: str = RESET_NOTICE) -> None:
        self.keyboard.hold_resets()
        self._started, self._label = time.monotonic(), label
        show = getattr(view, "show_notice", None)
        if callable(show):
            show(notice, wait_s=RESET_NOTICE_WAIT_S)
        print("{}: please wait, R is ignored until the scene is back ...".format(label))

    def settle(self, view: Any = None) -> dict[str, Any] | None:
        """After the first step on the new model: the log record (``None`` if nothing was held)."""
        if self._started is None:
            return None
        ignored = self.keyboard.release_resets(self.grace_s)
        show = getattr(view, "show_notice", None)
        if callable(show):
            show(None)
        now = time.monotonic()
        record = {
            "event": "reset_settled", "what": self._label, "held_s": now - self._started,
            "ignored_r_presses": ignored, "r_grace_s": self.grace_s, "timestamp_monotonic_s": now,
        }
        self._started = None
        print("{} done in {:.1f} s{}".format(
            self._label, record["held_s"], " ({} extra R ignored)".format(ignored) if ignored else ""))
        return record


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
