"""Threaded, native-resolution operator display for the LIBERO teleop entries.

Physics and control stay on the main thread. After every control step, and
once in the middle of it, the main thread only snapshots the three views into
MuJoCo scenes (``mjv_updateScene``: CPU only, ~0.1 ms for all three). A render
thread with its own OpenGL context draws the newest snapshot into one
offscreen frame at the window's real pixel size, adds the text and preview
overlays and shows it. MuJoCo and OpenCV release the GIL, so drawing and
display run alongside the next physics step instead of after it, and the
display gets two frames per 20 Hz control step.

The process is made DPI-aware and the window is shown 1:1 (no OpenCV or
Windows scaling), so what MuJoCo draws is exactly what reaches the screen.

Panels: the large main view on the left, two smaller views on the right.

* ``fixed``: a MuJoCo camera of the model (the main view, or the camera-only
  entry's orthographic plan / elevation cameras);
* ``follow``: a free camera that tracks the hand's grasp point from a fixed
  world direction and distance, so it always frames the fingers and whatever
  is in front of them, and never turns with the wrist (``V`` orbits the
  close-up around the hand).

Nothing here writes actions or physics state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional

import numpy as np

from .config import TriViewConfig

SLOTS = ("front", "wrist", "side")
GUTTER_PX = 6
MAIN_WIDTH_FRACTION = 2.0 / 3.0
SCENE_MAX_GEOMS = 2000
DETAIL_ORBIT_STEP_DEG = 90.0
PREVIEW_WIDTH_FRACTION = 0.26
WHITE = (240, 240, 240)
GREY = (160, 160, 160)
NOTICE_BGR = (0, 215, 255)  # amber; notices go on the BGR image right before imshow


@dataclass(frozen=True)
class FollowCamera:
    """A free camera aimed at the tracked point (plus ``lookat_offset_m``).

    MuJoCo convention: the camera looks along azimuth / elevation; azimuth 0
    looks along world +x, 90 along +y; negative elevation looks down.
    """

    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    fovy_deg: float = 45.0
    lookat_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orbit: bool = False  # ``V`` turns this one around the hand
    # Look-at low-pass time constant: the view glides after the hand instead
    # of shaking with it.
    smoothing_s: float = 0.10


@dataclass(frozen=True)
class Panel:
    slot: str  # "front" (main), "wrist" (upper right) or "side" (lower right)
    label: str
    camera: Optional[str] = None
    follow: Optional[FollowCamera] = None

    def __post_init__(self) -> None:
        if self.slot not in SLOTS or (self.camera is None) == (self.follow is None):
            raise ValueError("a panel needs a slot in {} and exactly one of camera / follow".format(SLOTS))


def enable_dpi_awareness() -> bool:
    """Ask Windows for real pixels (no bitmap stretching of our windows)."""

    if os.name != "nt":
        return False
    import ctypes

    try:
        return bool(ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)))  # per-monitor v2
    except Exception:
        try:
            return ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0
        except Exception:
            return False


def primary_work_area() -> Optional[tuple[int, int, int, int]]:
    """(x, y, width, height) of the primary monitor without the taskbar."""

    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    if not ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
        return None
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def default_view_size() -> tuple[int, int]:
    """Composite size that fills the primary monitor's work area 1:1."""

    area = primary_work_area()
    if area is None:
        return 1920, 1080
    _x, _y, width, height = area
    # Leave room for the window frame and title bar.
    return max(960, width - 16), max(540, height - 48)


def parse_view_size(text: str) -> tuple[int, int]:
    width, _, height = str(text).lower().partition("x")
    size = int(width), int(height)
    if not (640 <= size[0] <= 3840 and 360 <= size[1] <= 2160):
        raise ValueError("view size must be WIDTHxHEIGHT within 640x360 .. 3840x2160")
    return size


def layout(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """(x, y, width, height) of the three panels in a ``width`` x ``height`` composite."""

    main = int(round((width - GUTTER_PX) * MAIN_WIDTH_FRACTION))
    right = width - GUTTER_PX - main
    upper = (height - GUTTER_PX) // 2
    return {
        "front": (0, 0, main, height),
        "wrist": (main + GUTTER_PX, 0, right, upper),
        "side": (main + GUTTER_PX, upper + GUTTER_PX, right, height - GUTTER_PX - upper),
    }


def _text(cv2: Any, frame: np.ndarray, text: str, origin: tuple[int, int], scale: float, color=WHITE) -> None:
    """Text on a darkened box, readable on bright walls and the table."""

    thickness = max(1, int(round(scale * 1.6)))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    box = frame[max(y - height - 5, 0):max(y + baseline + 3, 0), max(x - 5, 0):max(x + width + 5, 0)]
    box[:] = (box * 0.4).astype(np.uint8)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_notice(cv2: Any, image: np.ndarray, text: str, rect: tuple[int, int, int, int], ui_scale: float,
                color: tuple[int, int, int] = NOTICE_BGR) -> None:
    """Centred notice on a dark band across ``rect``: the first line large, the rest smaller."""

    lines = [line for line in str(text).split("\n") if line.strip()]
    if not lines:
        return
    x0, y0, width, height = rect
    pad = max(4, int(round(24 * ui_scale)))
    gap = max(2, int(round(14 * ui_scale)))
    rows = []
    for index, line in enumerate(lines):
        scale = (1.6 if index == 0 else 0.8) * ui_scale
        thickness = max(1, int(round(scale * 2.0)))
        (text_w, text_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if text_w > width - 2 * pad:  # narrow window: shrink the line to fit
            scale *= max(width - 2 * pad, 1) / float(text_w)
            thickness = max(1, int(round(scale * 2.0)))
            (text_w, text_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        rows.append((line, scale, thickness, text_w, text_h, baseline))
    block = sum(row[4] + row[5] for row in rows) + gap * (len(rows) - 1)
    top = y0 + max(0, (height - block) // 2)
    band = image[max(top - pad, 0):min(top + block + pad, image.shape[0]), max(x0, 0):x0 + width]
    band[:] = (band * 0.3).astype(np.uint8)
    y = top
    for line, scale, thickness, text_w, text_h, baseline in rows:
        y += text_h
        cv2.putText(image, line, (x0 + (width - text_w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)
        y += baseline + gap


class _SceneSet:
    """One snapshot of the three panels, plus what the render thread needs to draw it."""

    def __init__(self, model: Any) -> None:
        import mujoco

        self.model = model
        self.scenes = {slot: mujoco.MjvScene(model, maxgeom=SCENE_MAX_GEOMS) for slot in SLOTS}
        self.generation = -1
        self.sim_time = 0.0
        self.status_text: Optional[str] = None
        self.captured_at = 0.0


class LiberoViewWindow:
    """Three synchronized views in one 1:1 window, drawn on a render thread.

    The interface follows :class:`~wrist_teleop.visualization.TriViewCompositor`
    where the LIBERO entries use it: ``configure_cameras``, ``update``,
    ``panel_rects``, ``save_last_frame``, ``close``, ``laser``, ``overlay``,
    plus ``show_notice`` (shown even while the main thread is blocked).
    ``update`` never blocks on rendering; it returns the latest display stats.
    """

    def __init__(
        self,
        config: TriViewConfig,
        panels: Sequence[Panel],
        *,
        size: tuple[int, int],
        show_window: bool = True,
        track_site: Optional[str] = None,
        key_handler: Optional[Callable[[int], Any]] = None,
        on_close: Optional[Callable[[], Any]] = None,
        laser: Any = None,
        overlay: Optional[Callable[[np.ndarray, dict[str, tuple[int, int, int, int]]], Any]] = None,
        preview: Optional[Callable[[], Optional[np.ndarray]]] = None,
        subframes: int = 2,
    ) -> None:
        if sorted(panel.slot for panel in panels) != sorted(SLOTS):
            raise ValueError("need exactly one panel per slot {}".format(SLOTS))
        self.width, self.height = int(size[0]), int(size[1])
        self.rects = layout(self.width, self.height)
        self.panels = {panel.slot: panel for panel in panels}
        self.config = config
        self.show_window = bool(show_window)
        self.track_site = track_site
        self.key_handler, self.on_close = key_handler, on_close
        self.laser, self.overlay, self.preview = laser, overlay, preview
        self.subframes = max(1, int(subframes))
        self.ui_scale = self.height / 1080.0
        self.orbit_deg = 0.0
        self.last_frame: Optional[np.ndarray] = None

        self._sim: Any = None
        self._vopt: Any = None
        self._site_id = -1
        self._lookat: dict[str, np.ndarray] = {}
        self._lookat_time: Optional[float] = None
        self._status_text: Optional[str] = None
        self._generation = 0
        self._substep = 0
        self._capture_substeps: set[int] = set()
        self._sets: list[_SceneSet] = []
        self._lock = threading.Condition()
        self._ready: Optional[int] = None
        self._rendering: Optional[int] = None
        self._stop = threading.Event()
        self._closed = False
        self._close_notified = False
        self._gl_context: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stats: dict[str, Any] = {"frames": 0, "dropped": 0}
        self._error: Optional[str] = None
        self._capture_ms = 0.0
        self._notice: Optional[str] = None
        self._notice_version = 0
        self._notice_shown = 0
        self.camera_setup: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def panel_sizes(self) -> dict[str, tuple[int, int]]:
        return {slot: (rect[2], rect[3]) for slot, rect in self.rects.items()}

    def panel_rects(self) -> dict[str, tuple[int, int, int, int]]:
        return dict(self.rects)

    def configure_cameras(self, env: Any) -> dict[str, Any]:
        """Bind to the current model (startup and after every hard reset)."""

        import mujoco

        sim = env.sim
        model = sim.model._model
        front = self.panels["front"].camera
        if front is not None:
            camera = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, front))
            if camera < 0:
                raise ValueError("main view camera does not exist: {!r}".format(front))
            if front == self.config.front_camera_name:
                model.cam_pos[camera] = np.asarray(self.config.front_position_m)
                model.cam_quat[camera] = np.asarray(self.config.front_quat_wxyz)
                model.cam_fovy[camera] = float(self.config.front_fovy_deg)
            sim.forward()
        for panel in self.panels.values():
            if panel.camera is not None and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, panel.camera) < 0:
                raise ValueError("view camera does not exist: {!r}".format(panel.camera))
        self._site_id = -1
        if self.track_site is not None:
            self._site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.track_site))
        if self._site_id < 0 and any(panel.follow is not None for panel in self.panels.values()):
            raise ValueError("follow views need a tracked site, got {!r}".format(self.track_site))
        self._vopt = mujoco.MjvOption()
        robosuite_context = getattr(sim, "_render_context_offscreen", None)
        source = getattr(robosuite_context, "vopt", None)
        if source is not None:
            # Same visible geom groups as robosuite's own renderer (no collision meshes).
            self._vopt.geomgroup[:] = source.geomgroup
            self._vopt.sitegroup[:] = source.sitegroup
        else:
            self._vopt.geomgroup[0] = 0
        with self._lock:
            while self._rendering is not None:
                self._lock.wait(0.05)
            self._ready = None
            self._generation += 1
            self._sets = [_SceneSet(model), _SceneSet(model)]
        self._sim = sim
        self._lookat, self._lookat_time = {}, None
        substeps = max(1, int(round(env.control_timestep / model.opt.timestep)))
        # The post-step snapshot comes from ``update``; these fill the gaps.
        self._capture_substeps = {int(round(substeps * k / self.subframes)) for k in range(1, self.subframes)}
        self._install_substep_hook(sim)
        self.camera_setup = {
            "size": [self.width, self.height], "panels": {
                slot: {"label": panel.label, "camera": panel.camera,
                       "follow": None if panel.follow is None else vars(panel.follow)}
                for slot, panel in self.panels.items()},
            "track_site": self.track_site, "subframes": self.subframes,
        }
        if self._thread is None:
            self._start(model)
        return dict(self.camera_setup)

    def _install_substep_hook(self, sim: Any) -> None:
        """Snapshot once mid-step, so the display runs at twice the control rate."""

        if self.subframes < 2 or getattr(sim.step, "_libero_views_hook", False):
            return
        original = sim.step

        def step(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._substep += 1
            if self._substep in self._capture_substeps:
                try:
                    if self.laser is not None:
                        self.laser.update()
                    self._capture(self._status_text)
                except Exception as exc:  # the display must never stop physics
                    self._error = "{}: {}".format(type(exc).__name__, exc)
            return result

        step._libero_views_hook = True  # type: ignore[attr-defined]
        sim.step = step

    def cycle_detail_view(self) -> float:
        """Orbit the close-up 90 deg around the hand; returns the new offset."""

        self.orbit_deg = (self.orbit_deg + DETAIL_ORBIT_STEP_DEG) % 360.0
        return self.orbit_deg

    # ------------------------------------------------------------ main thread
    def _follow_lookats(self, sim_time: float) -> dict[str, np.ndarray]:
        """Low-passed grasp point per follow panel (each has its own time constant)."""

        point = np.array(self._sim.data._data.site_xpos[self._site_id], dtype=np.float64)
        restart = self._lookat_time is None or sim_time < self._lookat_time
        dt = 0.0 if restart else sim_time - self._lookat_time
        for slot, panel in self.panels.items():
            if panel.follow is None:
                continue
            previous = self._lookat.get(slot)
            if restart or previous is None:
                self._lookat[slot] = point
            else:
                alpha = 1.0 - np.exp(-dt / max(panel.follow.smoothing_s, 1e-3))
                self._lookat[slot] = previous + alpha * (point - previous)
        self._lookat_time = sim_time
        return self._lookat

    def _capture(self, status_text: Optional[str]) -> bool:
        import mujoco

        started = time.perf_counter()
        with self._lock:
            if not self._sets or self._closed:
                return False
            index = 0 if self._rendering == 1 else 1 if self._rendering == 0 else (
                1 - self._ready if self._ready is not None else 0)
            if self._ready is not None:
                self._stats["dropped"] += 1
                if self._ready == index:
                    self._ready = None
            target = self._sets[index]
            generation = self._generation
        model, data = self._sim.model._model, self._sim.data._data
        sim_time = float(data.time)
        lookats = self._follow_lookats(sim_time) if self._site_id >= 0 else {}
        perturb = mujoco.MjvPerturb()
        global_fovy = float(model.vis.global_.fovy)
        try:
            for slot, panel in self.panels.items():
                camera = mujoco.MjvCamera()
                if panel.camera is not None:
                    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    camera.fixedcamid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, panel.camera))
                else:
                    follow = panel.follow
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = lookats[slot] + np.asarray(follow.lookat_offset_m)
                    camera.distance = follow.distance_m
                    camera.azimuth = follow.azimuth_deg + (self.orbit_deg if follow.orbit else 0.0)
                    camera.elevation = follow.elevation_deg
                    model.vis.global_.fovy = follow.fovy_deg
                if self.laser is not None:
                    self.laser.set_render_view(slot)
                mujoco.mjv_updateScene(model, data, self._vopt, perturb, camera,
                                       mujoco.mjtCatBit.mjCAT_ALL, target.scenes[slot])
                model.vis.global_.fovy = global_fovy
        finally:
            model.vis.global_.fovy = global_fovy
            if self.laser is not None:
                self.laser.restore_render_visibility()
        target.generation, target.sim_time = generation, sim_time
        target.status_text, target.captured_at = status_text, time.perf_counter()
        with self._lock:
            if generation == self._generation:
                self._ready = index
                self._lock.notify_all()
        self._capture_ms = 1e3 * (time.perf_counter() - started)
        return True

    def update(self, env: Any, status_text: Optional[str] = None) -> dict[str, Any]:
        """Snapshot the post-step state for display; never waits for drawing."""

        self._substep = 0
        self._status_text = status_text
        if self._closed:
            return {"rendered": False, "closed": True}
        captured = False
        try:
            captured = self._capture(status_text)
        except Exception as exc:
            self._error = "{}: {}".format(type(exc).__name__, exc)
        with self._lock:
            stats = dict(self._stats)
        record = {
            "rendered": captured, "closed": self._closed,
            "sync_time_s": float(self._sim.data.time) if self._sim is not None else None,
            "capture_ms": self._capture_ms, "display": stats, "orbit_deg": self.orbit_deg,
        }
        if self._error is not None:
            record["error"] = self._error
        return record

    def display_stats(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def show_notice(self, text: Optional[str], wait_s: float = 0.0) -> bool:
        """Big notice across the main view; ``None`` removes it.

        The render thread puts it over the last frame at once, so it shows
        even while the main thread is blocked (a hard reset). ``wait_s``: wait
        at most that long until it is on screen. Returns whether it got there
        in time (``False`` without a window).
        """

        with self._lock:
            self._notice = str(text) if text else None
            self._notice_version += 1
            version = self._notice_version
            self._lock.notify_all()
            if wait_s <= 0.0 or not self.show_window or self._thread is None or self._closed:
                return False
            deadline = time.perf_counter() + float(wait_s)
            while self._notice_shown < version and not self._closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self._lock.wait(remaining)
            return self._notice_shown >= version

    def save_last_frame(self, path: str | Path) -> bool:
        frame = self.last_frame
        if frame is None:
            return False
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            return bool(cv2.imwrite(str(target), frame[..., ::-1]))
        except Exception:
            return False

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._gl_context is not None:
            try:
                self._gl_context.free()  # GLFW windows are destroyed on the main thread
            except Exception:
                pass
            self._gl_context = None
        self._closed = True

    # ---------------------------------------------------------- render thread
    def _start(self, model: Any) -> None:
        from mujoco.glfw import GLContext

        # GLFW creates windows only on the main thread; the context is then
        # made current on the render thread, which owns it from there on.
        self._gl_context = GLContext(self.width, self.height)
        self._thread = threading.Thread(target=self._run, name="libero-views", daemon=True)
        self._thread.start()

    def _notify_closed(self) -> None:
        self._closed = True
        if not self._close_notified:
            self._close_notified = True
            if self.on_close is not None:
                self.on_close()

    def _run(self) -> None:
        import cv2
        import mujoco

        context = None
        context_model = None
        window_created = False
        buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        full = mujoco.MjrRect(0, 0, self.width, self.height)
        shown_at: list[float] = []
        try:
            self._gl_context.make_current()
            while not self._stop.is_set():
                with self._lock:
                    if self._ready is None:
                        self._lock.wait(0.03)
                    index = self._ready
                    if index is not None:
                        self._ready, self._rendering = None, index
                if index is not None:
                    scene_set = self._sets[index]
                    started = time.perf_counter()
                    try:
                        if context is None or context_model is not scene_set.model:
                            if context is not None:
                                context.free()
                            scene_set.model.vis.global_.offwidth = max(self.width, scene_set.model.vis.global_.offwidth)
                            scene_set.model.vis.global_.offheight = max(self.height, scene_set.model.vis.global_.offheight)
                            context = mujoco.MjrContext(scene_set.model, mujoco.mjtFontScale.mjFONTSCALE_150)
                            context_model = scene_set.model
                        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
                        mujoco.mjr_rectangle(full, 0.0, 0.0, 0.0, 1.0)
                        for slot, (x, y, width, height) in self.rects.items():
                            # OpenGL viewports start at the bottom-left corner.
                            viewport = mujoco.MjrRect(x, self.height - y - height, width, height)
                            mujoco.mjr_render(viewport, scene_set.scenes[slot], context)
                        mujoco.mjr_readPixels(buffer, None, full, context)
                        status_text, sim_time, captured_at = scene_set.status_text, scene_set.sim_time, scene_set.captured_at
                    finally:
                        with self._lock:
                            self._rendering = None
                            self._lock.notify_all()
                    drawn = time.perf_counter()
                    frame = cv2.flip(buffer, 0)
                    self._decorate(cv2, frame, status_text)
                    self.last_frame = frame
                    if self.show_window and not self._closed:
                        if not window_created:
                            cv2.namedWindow(self.config.window_name, cv2.WINDOW_AUTOSIZE)
                            area = primary_work_area()
                            if area is not None:
                                cv2.moveWindow(self.config.window_name, area[0], area[1])
                            window_created = True
                        self._show(cv2, frame)
                        self._poll_window(cv2)
                    finished = time.perf_counter()
                    shown_at = [t for t in shown_at if finished - t < 1.0] + [finished]
                    with self._lock:
                        self._stats.update({
                            "frames": self._stats["frames"] + 1, "fps": float(len(shown_at)),
                            "draw_ms": 1e3 * (drawn - started), "show_ms": 1e3 * (finished - drawn),
                            "latency_ms": 1e3 * (finished - captured_at), "sim_time_s": sim_time,
                        })
                elif window_created and not self._closed:
                    with self._lock:  # a changed notice goes over the frozen picture now
                        redraw = self._ready is None and self._notice_shown != self._notice_version
                    if redraw and self.last_frame is not None:
                        self._show(cv2, self.last_frame)
                    self._poll_window(cv2)
        except Exception as exc:
            self._error = "render thread: {}: {}".format(type(exc).__name__, exc)
        finally:
            if context is not None:
                try:
                    context.free()
                except Exception:
                    pass
            try:
                import glfw

                glfw.make_context_current(None)
            except Exception:
                pass
            if window_created:
                try:
                    cv2.destroyWindow(self.config.window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass

    def _show(self, cv2: Any, frame: np.ndarray) -> None:
        """imshow ``frame`` (RGB, left unchanged) with the current notice on top."""

        with self._lock:
            notice, version = self._notice, self._notice_version
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if notice:
            draw_notice(cv2, image, notice, self.rects["front"], self.ui_scale)
        cv2.imshow(self.config.window_name, image)
        with self._lock:
            self._notice_shown = max(self._notice_shown, version)
            self._lock.notify_all()

    def _poll_window(self, cv2: Any) -> None:
        raw = int(cv2.waitKey(1))
        if raw >= 0 and self.key_handler is not None:
            try:
                self.key_handler(raw & 0xFF)
            except Exception as exc:
                self._error = "key handler: {}: {}".format(type(exc).__name__, exc)
        try:
            if float(cv2.getWindowProperty(self.config.window_name, cv2.WND_PROP_VISIBLE)) < 1.0:
                self._notify_closed()
        except Exception:
            pass

    def _decorate(self, cv2: Any, frame: np.ndarray, status_text: Optional[str]) -> None:
        scale = 0.6 * self.ui_scale
        pad = int(round(14 * self.ui_scale))
        line = int(round(30 * self.ui_scale))
        if status_text:
            _text(cv2, frame, str(status_text), (pad, line), scale)
        for slot, panel in self.panels.items():
            x, y, _width, _height = self.rects[slot]
            if slot != "front" and panel.label:
                label = panel.label
                if panel.follow is not None and panel.follow.orbit:
                    label = "{}  [V] orbit".format(label)
                _text(cv2, frame, label, (x + pad, y + line), scale)
        if self.preview is not None:
            try:
                camera = self.preview()
            except Exception:
                camera = None
            if camera is not None:
                x0, y0, width, height = self.rects["front"]
                inset_w = int(round(width * PREVIEW_WIDTH_FRACTION))
                inset_h = int(round(inset_w * camera.shape[0] / camera.shape[1]))
                inset = cv2.resize(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB), (inset_w, inset_h),
                                   interpolation=cv2.INTER_AREA)
                left, top = x0 + pad, y0 + height - inset_h - pad
                frame[top:top + inset_h, left:left + inset_w] = inset
                cv2.rectangle(frame, (left - 1, top - 1), (left + inset_w, top + inset_h), WHITE, 1)
        if self.overlay is not None:
            try:
                self.overlay(frame, self.panel_rects())
            except Exception as exc:
                self._error = "overlay: {}: {}".format(type(exc).__name__, exc)
