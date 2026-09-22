"""Reusable, nonphysical visual assistance for robosuite teleoperation.

The module deliberately keeps rendering separate from control and physics:

* :class:`TriViewCompositor` renders three cameras after an ``env.step`` from
  the same MuJoCo state, then places them in one optional OpenCV window.
* :class:`LaserPointer` repurposes an existing transparent MuJoCo *site* as a
  TCP-attached cylinder.  A site has no mass, collision, contact, actuator,
  or qpos/qvel contribution.

Neither class sleeps, steps an environment, writes actions, or owns a native
MuJoCo viewer.  This lets callers keep the existing 20 Hz control loop and
route R/Q keys through their established safety handler.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import LaserConfig, TriViewConfig

__all__ = ["TriViewCompositor", "LaserPointer", "LaserConfig", "TriViewConfig"]

try:  # Keep compositing importable for configuration-only / synthetic tests.
    import mujoco
except ModuleNotFoundError:  # pragma: no cover - the project lock supplies MuJoCo
    mujoco = None  # type: ignore[assignment]


_EPSILON_M = 1.0e-5


def _sim_from(env: Any) -> Any:
    """Accept either a robosuite environment or a light ``sim`` test object."""
    return getattr(env, "sim", env)


def _native_model(model: Any) -> Any:
    """Return MuJoCo's native model from robosuite's binding wrapper."""
    return getattr(model, "_model", model)


def _native_data(data: Any) -> Any:
    """Return MuJoCo's native data from robosuite's binding wrapper."""
    return getattr(data, "_data", data)


def _json_float(value: Any) -> float | None:
    """Convert finite scalar telemetry to JSON-safe float, otherwise ``None``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _camera_frames(config: TriViewConfig) -> tuple[tuple[str, str], ...]:
    return (
        ("front", config.front_camera_name),
        ("wrist", config.wrist_camera_name),
        ("side", config.side_camera_name),
    )


class TriViewCompositor:
    """Render a synchronized front / wrist / side composite without stepping.

    ``update(env, status_text=...)`` is intended immediately after the
    caller's existing ``env.step(action)``.  It returns a compact JSON-ready
    record so callers can log the exact simulation timestamp shared by all
    images.  Set ``show_window=False`` (or configure it that way) for
    headless runs; offscreen rendering still works and no HighGUI calls occur.

    ``key_handler`` receives OpenCV key codes.  ``on_close`` is called once if
    the user closes the composite window.  Both are intentionally callbacks so
    this module cannot bypass the application's existing safe R/Q shutdown.
    """

    def __init__(
        self,
        config: TriViewConfig,
        *,
        show_window: bool | None = None,
        key_handler: Callable[[int], Any] | None = None,
        on_close: Callable[[], Any] | None = None,
        laser: LaserPointer | None = None,
        clock: Callable[[], float] = time.monotonic,
        cv2_module: Any | None = None,
    ) -> None:
        if not isinstance(config, TriViewConfig):
            raise TypeError("config must be a validated TriViewConfig")
        self.config = config
        self.show_window = config.show_window if show_window is None else bool(show_window)
        self.key_handler = key_handler
        self.on_close = on_close
        self.laser = laser
        self._clock = clock
        self._cv2 = cv2_module
        self._window_created = False
        self._closed = False
        self._close_notified = False
        self._last_render_at: float | None = None
        self._configured_model_id: int | None = None
        self._camera_setup: dict[str, Any] = {}
        self.last_frame: np.ndarray | None = None
        self.last_frames: dict[str, np.ndarray] = {}

    @property
    def closed(self) -> bool:
        return self._closed

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    @staticmethod
    def _state_time(sim: Any) -> float | None:
        data = getattr(sim, "data", None)
        return _json_float(getattr(data, "time", None))

    def _base_record(self, sim: Any, *, rendered: bool) -> dict[str, Any]:
        sync_time = self._state_time(sim)
        return {
            "rendered": bool(rendered),
            "closed": bool(self._closed),
            "sync_time_s": sync_time,
            "camera_names": {view: name for view, name in _camera_frames(self.config)},
            "camera_configuration": dict(self._camera_setup),
            "key": None,
        }

    @staticmethod
    def _camera_id(model: Any, camera_name: str) -> int:
        resolver = getattr(model, "camera_name2id", None)
        if callable(resolver):
            return int(resolver(camera_name))
        native = _native_model(model)
        if mujoco is None:
            return -1
        return int(mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))

    def _apply_camera_configuration(self, sim: Any) -> None:
        """Apply the configured fixed-camera poses once for this MuJoCo model.

        `front` and `side` are world cameras. `wrist` stays attached to the
        hand body, so only its FOV is changed. This method writes camera
        visual settings only; it does not step physics or change robot state.
        """
        model = getattr(sim, "model", None)
        if model is None or id(model) == self._configured_model_id:
            return
        setup: dict[str, Any] = {}
        for view, camera_name in _camera_frames(self.config):
            camera_id = self._camera_id(model, camera_name)
            if camera_id < 0:
                raise ValueError(f"configured MuJoCo camera does not exist: {camera_name!r}")
            override = self.config.fixed_camera_overrides[view]
            position = override["position_m"]
            quaternion = override["quat_wxyz"]
            if position is not None:
                model.cam_pos[camera_id] = np.asarray(position, dtype=np.float64)
            if quaternion is not None:
                model.cam_quat[camera_id] = np.asarray(quaternion, dtype=np.float64)
            model.cam_fovy[camera_id] = float(override["fovy_deg"])
            setup[view] = {
                "camera_name": camera_name,
                "camera_id": camera_id,
                "position_m": None if position is None else list(position),
                "quat_wxyz": None if quaternion is None else list(quaternion),
                "fovy_deg": float(override["fovy_deg"]),
            }
        # cam_xmat is the authoritative world-from-camera rotation after the
        # model overrides above.  Recompute it once without advancing physics
        # so both display telemetry and camera-relative control can use the
        # same real extrinsics.
        forward = getattr(sim, "forward", None)
        if callable(forward):
            forward()
        data = getattr(sim, "data", None)
        native_data = _native_data(data) if data is not None else None
        matrices = getattr(native_data, "cam_xmat", None)
        if matrices is not None:
            for view, entry in setup.items():
                matrix = np.asarray(matrices[int(entry["camera_id"])], dtype=np.float64).reshape(3, 3)
                if np.all(np.isfinite(matrix)):
                    entry["world_from_camera_rotation"] = matrix.tolist()
        self._camera_setup = setup
        self._configured_model_id = id(model)

    def configure_cameras(self, env: Any) -> dict[str, Any]:
        """Apply fixed camera poses without rendering or opening a window.

        Entrypoints use this at startup and after ``R`` resets, then bind
        :class:`~wrist_teleop.mapper.SixDofMapper` to the resulting live
        ``cam_xmat``.  It changes visual camera metadata only and does not
        step the environment or create a competing native viewer.
        """
        sim = _sim_from(env)
        self._apply_camera_configuration(sim)
        return dict(self._camera_setup)

    @staticmethod
    def _normalise_image(image: Any, *, width: int, height: int, camera_name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[0] != height or array.shape[1] != width or array.shape[2] < 3:
            raise ValueError(
                f"camera {camera_name!r} returned {array.shape}, expected ({height}, {width}, 3+)"
            )
        array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        # MuJoCo's offscreen buffer uses a bottom-left origin. Convert it once
        # here so every panel, text overlay, PNG, and OpenCV window is upright.
        return np.ascontiguousarray(array[::-1])

    def _resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        cv2 = self._load_cv2()
        if image.shape[:2] == (int(height), int(width)):
            return image
        # This is only a defensive fallback for a third-party simulator that
        # ignores the requested output dimensions. Normal MuJoCo rendering
        # reaches its final panel size directly and takes this path zero times.
        return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    def _compose(self, frames: dict[str, np.ndarray], status_text: str | None) -> np.ndarray:
        front, wrist, side = (frames[name] for name in ("front", "wrist", "side"))
        height, width = self.config.image_height, self.config.image_width
        gutter = self.config.gutter_px
        panel_sizes = self.config.panel_sizes
        if self.config.layout == "main_left":
            thumb_width, thumb_height = panel_sizes["wrist"]
            _, side_height = panel_sizes["side"]
            canvas = np.zeros((height, width + gutter + thumb_width, 3), dtype=np.uint8)
            canvas[:, :width] = front
            canvas[:thumb_height, width + gutter:] = self._resize(wrist, thumb_width, thumb_height)
            canvas[thumb_height + gutter:, width + gutter:] = self._resize(
                side, thumb_width, side_height
            )
        else:  # TriViewConfig validates this to ``main_top``.
            thumb_width, thumb_height = panel_sizes["wrist"]
            side_width, _ = panel_sizes["side"]
            canvas = np.zeros((height + gutter + thumb_height, width, 3), dtype=np.uint8)
            canvas[:height, :] = front
            canvas[height + gutter:, :thumb_width] = self._resize(wrist, thumb_width, thumb_height)
            canvas[height + gutter:, thumb_width + gutter:] = self._resize(
                side, side_width, thumb_height
            )
        if status_text:
            try:
                cv2 = self._load_cv2()
                cv2.putText(
                    canvas,
                    str(status_text),
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                # Text is optional visual polish; a missing HighGUI / OpenCV
                # feature must not prevent the simulation control loop.
                pass
        return canvas

    def _notify_closed(self) -> None:
        if self._close_notified:
            return
        self._close_notified = True
        if self.on_close is not None:
            self.on_close()

    def _show(self, frame: np.ndarray) -> tuple[int | None, Any | None, str | None]:
        """Display one frame and return key / routed action / GUI error."""
        if not self.show_window:
            return None, None, None
        try:
            cv2 = self._load_cv2()
            if not self._window_created:
                cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    self.config.window_name,
                    int(self.config.display_width or frame.shape[1]),
                    int(self.config.display_height or frame.shape[0]),
                )
                self._window_created = True
            # MuJoCo offscreen render output is RGB while HighGUI expects BGR.
            cv2.imshow(self.config.window_name, frame[..., ::-1])
            raw_key = int(cv2.waitKey(1))
            key = None if raw_key < 0 else raw_key & 0xFF
            action = self.key_handler(key) if key is not None and self.key_handler is not None else None
            try:
                visible = float(cv2.getWindowProperty(self.config.window_name, cv2.WND_PROP_VISIBLE))
                if visible < 1.0:
                    self._closed = True
                    self._notify_closed()
            except Exception:
                # Some OpenCV backends do not expose WND_PROP_VISIBLE.  In
                # that case waitKey still handles input and explicit close().
                pass
            return key, action, None
        except Exception as exc:  # A GUI backend failure must be visible in logs, not crash control.
            self._closed = True
            self._notify_closed()
            return None, None, f"{type(exc).__name__}: {exc}"

    def update(self, env: Any, status_text: str | None = None) -> dict[str, Any]:
        """Render one same-state composite, or cheaply report a rate-limited tick."""
        sim = _sim_from(env)
        if self._closed:
            return self._base_record(sim, rendered=False)
        if not self.config.enabled:
            record = self._base_record(sim, rendered=False)
            record["disabled"] = True
            return record
        now = float(self._clock())
        interval = 1.0 / self.config.render_hz
        if self._last_render_at is not None and now - self._last_render_at < interval:
            record = self._base_record(sim, rendered=False)
            record["rate_limited"] = True
            return record

        try:
            self._apply_camera_configuration(sim)
        except Exception as exc:
            # A bad optional view must never stop the arm control loop. The
            # subsequent render still gets a precise logged configuration error.
            self._camera_setup = {"error": f"{type(exc).__name__}: {exc}"}

        before_time = self._state_time(sim)
        frames: dict[str, np.ndarray] = {}
        panel_sizes = self.config.panel_sizes
        try:
            for view, camera_name in _camera_frames(self.config):
                if self.laser is not None:
                    self.laser.set_render_view(view)
                width, height = panel_sizes[view]
                image = sim.render(
                    width=width,
                    height=height,
                    camera_name=camera_name,
                )
                frames[view] = self._normalise_image(
                    image,
                    width=width,
                    height=height,
                    camera_name=camera_name,
                )
        except Exception as exc:
            if self.laser is not None:
                self.laser.restore_render_visibility()
            record = self._base_record(sim, rendered=False)
            record["render_error"] = f"{type(exc).__name__}: {exc}"
            return record
        finally:
            if self.laser is not None:
                self.laser.restore_render_visibility()

        after_time = self._state_time(sim)
        prior_render_at = self._last_render_at
        resize_operations = sum(
            int(frames[view].shape[:2] != (int(size[1]), int(size[0])))
            for view, size in panel_sizes.items()
        )
        frame = self._compose(frames, status_text)
        self.last_frames = {name: image.copy() for name, image in frames.items()}
        self.last_frame = frame.copy()
        self._last_render_at = now
        key, action, gui_error = self._show(frame)
        record = self._base_record(sim, rendered=True)
        record.update(
            {
                "state_time_before_s": before_time,
                "state_time_after_s": after_time,
                "same_state": before_time == after_time,
                "resolution": [self.config.image_width, self.config.image_height],
                "native_panel_resolutions": {
                    view: [int(size[0]), int(size[1])] for view, size in panel_sizes.items()
                },
                "composite_shape": [int(value) for value in frame.shape],
                "composite_resolution": [int(frame.shape[1]), int(frame.shape[0])],
                "display_resolution": [
                    int(self.config.display_width or frame.shape[1]),
                    int(self.config.display_height or frame.shape[0]),
                ],
                "resize_operations": int(resize_operations),
                "render_interval_s": None if prior_render_at is None else float(now - prior_render_at),
                "actual_render_hz": (
                    None
                    if prior_render_at is None or now <= prior_render_at
                    else float(1.0 / (now - prior_render_at))
                ),
                "key": key,
            }
        )
        if action is not None:
            record["key_action"] = action
        if gui_error is not None:
            record["gui_error"] = gui_error
        record["closed"] = bool(self._closed)
        return record

    def close(self) -> None:
        """Release only this optional HighGUI window; never close the environment."""
        if self._window_created:
            try:
                self._load_cv2().destroyWindow(self.config.window_name)
            except Exception:
                pass
        self._window_created = False
        self._closed = True

    def save_last_frame(self, path: str | Path) -> bool:
        """Write the latest composite image without stepping or changing state.

        Entrypoints use this once per run as lightweight evidence that the
        synchronized visual path actually rendered. The conversion is only
        for OpenCV's BGR file writer; ``last_frame`` remains RGB.
        """
        if self.last_frame is None:
            return False
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            cv2 = self._load_cv2()
            return bool(cv2.imwrite(str(target), self.last_frame[..., ::-1]))
        except Exception:
            return False


class LaserPointer:
    """Draw a TCP-attached MuJoCo site cylinder without changing physics.

    The pointer resolves names with an exact match first and then a unique
    robosuite-prefixed suffix match.  This lets the shared default names
    ``grip_site`` and ``grip_site_cylinder`` work for the generated Allegro
    gripper while allowing explicit names for other grippers.  Missing sites
    simply disable the feature and return a diagnostic record.
    """

    def __init__(self, env: Any, config: LaserConfig) -> None:
        if not isinstance(config, LaserConfig):
            raise TypeError("config must be a validated LaserConfig")
        self.env = env
        self.config = config
        self.sim = _sim_from(env)
        self.model = getattr(self.sim, "model", None)
        self.data = getattr(self.sim, "data", None)
        self._native_model = _native_model(self.model) if self.model is not None else None
        self._native_data = _native_data(self.data) if self.data is not None else None
        self._start_site_id = -1
        self._visual_site_id = -1
        self._start_site_name: str | None = None
        self._visual_site_name: str | None = None
        self._visual_body_id = -1
        self._excluded_geom_ids: set[int] = set()
        self._original: dict[str, np.ndarray] = {}
        self._available = False
        self._active = False
        self._closed = False
        self._reason: str | None = None
        self.last_update: dict[str, Any] | None = None
        self._initialize()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def active(self) -> bool:
        return self._active

    @staticmethod
    def _object_name(native_model: Any, object_type: Any, index: int) -> str | None:
        if mujoco is None:
            return None
        name = mujoco.mj_id2name(native_model, object_type, int(index))
        return str(name) if name is not None else None

    def _find_site(self, configured_name: str) -> tuple[int, str | None]:
        if mujoco is None or self._native_model is None:
            return -1, None
        exact = int(
            mujoco.mj_name2id(
                self._native_model, mujoco.mjtObj.mjOBJ_SITE, configured_name
            )
        )
        if exact >= 0:
            return exact, configured_name
        suffix = f"_{configured_name}"
        matches = [
            (index, self._object_name(self._native_model, mujoco.mjtObj.mjOBJ_SITE, index))
            for index in range(int(self._native_model.nsite))
        ]
        matches = [(index, name) for index, name in matches if name is not None and name.endswith(suffix)]
        if len(matches) == 1:
            return int(matches[0][0]), matches[0][1]
        return -1, None

    def _robot_geom_ids(self, start_site_id: int) -> set[int]:
        """Find geometry in the robot tree that must not shorten the laser."""
        if self.model is None or self._native_model is None:
            return set()
        start_body = int(self.model.site_bodyid[start_site_id])
        root = start_body
        # The world body is 0.  Stop at its first child, which is typically
        # robot0_base, so table and objects remain eligible ray targets.
        while int(self.model.body_parentid[root]) not in (0, -1):
            root = int(self.model.body_parentid[root])
        if root == 0:
            return set()

        def belongs_to_root(body_id: int) -> bool:
            current = int(body_id)
            while current not in (0, -1):
                if current == root:
                    return True
                current = int(self.model.body_parentid[current])
            return current == root

        return {
            geom_id
            for geom_id in range(int(self._native_model.ngeom))
            if belongs_to_root(int(self.model.geom_bodyid[geom_id]))
        }

    def _initialize(self) -> None:
        if not self.config.enabled:
            self._reason = "disabled_by_config"
            return
        if mujoco is None:
            self._reason = "mujoco_unavailable"
            return
        if self.model is None or self.data is None or self._native_model is None or self._native_data is None:
            self._reason = "environment_has_no_mujoco_sim"
            return
        self._start_site_id, self._start_site_name = self._find_site(self.config.start_site)
        self._visual_site_id, self._visual_site_name = self._find_site(self.config.visual_site)
        if self._start_site_id < 0:
            self._reason = f"missing_start_site:{self.config.start_site}"
            return
        if self._visual_site_id < 0:
            self._reason = f"missing_visual_site:{self.config.visual_site}"
            return
        try:
            self._visual_body_id = int(self.model.site_bodyid[self._visual_site_id])
            self._original = {
                "site_pos": np.asarray(self.model.site_pos[self._visual_site_id], dtype=np.float64).copy(),
                "site_quat": np.asarray(self.model.site_quat[self._visual_site_id], dtype=np.float64).copy(),
                "site_size": np.asarray(self.model.site_size[self._visual_site_id], dtype=np.float64).copy(),
                "site_rgba": np.asarray(self.model.site_rgba[self._visual_site_id], dtype=np.float64).copy(),
                # MuJoCo compiles a zero-offset, identity site as
                # ``sameframe``.  Its fast path ignores runtime site_pos /
                # site_quat, so preserve and temporarily clear this *site
                # visual transform* flag when reusing the zero-offset laser.
                "site_sameframe": np.asarray(self.model.site_sameframe[self._visual_site_id]).copy(),
            }
            self._excluded_geom_ids = self._robot_geom_ids(self._start_site_id)
        except Exception as exc:
            self._reason = f"site_setup_failed:{type(exc).__name__}:{exc}"
            return
        self._available = True
        self._reason = None

    def _forward(self) -> None:
        forward = getattr(self.sim, "forward", None)
        if callable(forward):
            forward()
        elif mujoco is not None and self._native_model is not None and self._native_data is not None:
            mujoco.mj_forward(self._native_model, self._native_data)

    def _ray_to_scene(self, start: np.ndarray, direction: np.ndarray) -> tuple[float, int | None, list[int]]:
        """Return first non-robot ray hit, advancing past self geometry."""
        assert mujoco is not None and self._native_model is not None and self._native_data is not None
        origin = np.asarray(start, dtype=np.float64).copy()
        travelled = 0.0
        skipped: list[int] = []
        max_iterations = max(8, int(self._native_model.ngeom) + 2)
        for _ in range(max_iterations):
            geom_id = np.full(1, -1, dtype=np.int32)
            distance = float(
                mujoco.mj_ray(
                    self._native_model,
                    self._native_data,
                    origin,
                    direction,
                    None,
                    1,
                    -1,
                    geom_id,
                )
            )
            if not np.isfinite(distance) or distance < 0.0:
                return self.config.max_length_m, None, skipped
            hit_distance = travelled + distance
            if hit_distance >= self.config.max_length_m:
                return self.config.max_length_m, None, skipped
            hit_id = int(geom_id[0])
            if hit_id not in self._excluded_geom_ids:
                return max(0.0, hit_distance), hit_id, skipped
            skipped.append(hit_id)
            advance = max(distance + _EPSILON_M, _EPSILON_M)
            travelled += advance
            if travelled >= self.config.max_length_m:
                return self.config.max_length_m, None, skipped
            origin = origin + direction * advance
        return self.config.max_length_m, None, skipped

    def _set_cylinder_pose(self, start: np.ndarray, direction: np.ndarray, length: float) -> None:
        """Place the visual cylinder midpoint and align its local +Z to ray."""
        assert mujoco is not None and self.model is not None and self.data is not None
        midpoint = start + direction * (0.5 * length)
        body_position = np.asarray(self.data.xpos[self._visual_body_id], dtype=np.float64)
        body_rotation = np.asarray(self.data.xmat[self._visual_body_id], dtype=np.float64).reshape(3, 3)
        local_position = body_rotation.T @ (midpoint - body_position)
        world_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_quatZ2Vec(world_quaternion, direction)
        world_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(world_rotation, world_quaternion)
        local_rotation = body_rotation.T @ world_rotation.reshape(3, 3)
        local_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(local_quaternion, local_rotation.reshape(9))

        size = self._original["site_size"].copy()
        size[0] = self.config.width_m
        if size.size > 1:
            size[1] = 0.5 * length
        color = np.asarray(self.config.color_rgba, dtype=np.float64)
        self.model.site_pos[self._visual_site_id] = local_position
        self.model.site_quat[self._visual_site_id] = local_quaternion
        self.model.site_size[self._visual_site_id] = size
        self.model.site_rgba[self._visual_site_id] = color
        self.model.site_sameframe[self._visual_site_id] = 0
        self._forward()

    def _record(self, *, active: bool, **extra: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "enabled": bool(self.config.enabled),
            "available": bool(self._available),
            "active": bool(active),
            "start_site": self._start_site_name or self.config.start_site,
            "visual_site": self._visual_site_name or self.config.visual_site,
        }
        if self._reason is not None:
            record["reason"] = self._reason
        record.update(extra)
        self.last_update = record
        return record

    def update(self) -> dict[str, Any]:
        """Update only visual site fields and return JSON-ready ray telemetry."""
        if self._closed:
            return self._record(active=False, reason="closed")
        if not self._available:
            return self._record(active=False)
        assert self.data is not None and self.model is not None and mujoco is not None
        start = np.asarray(self.data.site_xpos[self._start_site_id], dtype=np.float64).copy()
        start_rotation = np.asarray(self.data.site_xmat[self._start_site_id], dtype=np.float64).reshape(3, 3)
        if self.config.direction_frame == "world":
            # A world-aligned ray remains vertical even when the hand rotates.
            # This is used by Panda + Allegro's palm-to-table aid.
            direction = np.asarray(self.config.world_direction, dtype=np.float64)
        else:
            direction = start_rotation @ np.asarray(self.config.local_direction, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(direction)) or direction_norm <= 1.0e-12:
            self.restore()
            return self._record(active=False, reason="invalid_tcp_pose")
        direction = direction / direction_norm
        length, hit_id, skipped = self._ray_to_scene(start, direction)
        if hit_id is None and self.config.hide_on_no_hit:
            self.model.site_rgba[self._visual_site_id] = np.array(
                (*self.config.color_rgba[:3], 0.0), dtype=np.float64
            )
            self._forward()
            self._active = False
            return self._record(
                active=False,
                origin_world_m=start.tolist(),
                direction_world=direction.tolist(),
                length_m=0.0,
                hit_geom=None,
                skipped_geom_ids=skipped,
            )
        self._set_cylinder_pose(start, direction, length)
        self._active = True
        hit_name = self._object_name(self._native_model, mujoco.mjtObj.mjOBJ_GEOM, hit_id) if hit_id is not None else None
        visual_center = np.asarray(self.data.site_xpos[self._visual_site_id], dtype=np.float64).copy()
        visual_axis = np.asarray(self.data.site_xmat[self._visual_site_id], dtype=np.float64).reshape(3, 3)[:, 2].copy()
        return self._record(
            active=True,
            origin_world_m=start.tolist(),
            direction_world=direction.tolist(),
            direction_frame=self.config.direction_frame,
            length_m=float(length),
            hit_geom=hit_name,
            hit_geom_id=hit_id,
            skipped_geom_ids=skipped,
            visual_center_world_m=visual_center.tolist(),
            visual_axis_world=visual_axis.tolist(),
        )

    def set_render_view(self, view: str) -> bool:
        """Temporarily hide the visual site from a configured compositor panel."""
        if not self._available or not self._active:
            return False
        color = np.asarray(self.config.color_rgba, dtype=np.float64)
        if view not in self.config.visible_views:
            color[3] = 0.0
        assert self.model is not None
        self.model.site_rgba[self._visual_site_id] = color
        return bool(color[3] > 0.0)

    def restore_render_visibility(self) -> None:
        """Return per-render alpha changes to the configured laser opacity."""
        if not self._available or not self._active:
            return
        assert self.model is not None
        self.model.site_rgba[self._visual_site_id] = np.asarray(self.config.color_rgba, dtype=np.float64)

    def restore(self) -> bool:
        """Restore every changed visual field exactly; physics arrays are untouched."""
        if not self._available or not self._original or self.model is None:
            self._active = False
            return False
        self.model.site_pos[self._visual_site_id] = self._original["site_pos"]
        self.model.site_quat[self._visual_site_id] = self._original["site_quat"]
        self.model.site_size[self._visual_site_id] = self._original["site_size"]
        self.model.site_rgba[self._visual_site_id] = self._original["site_rgba"]
        self.model.site_sameframe[self._visual_site_id] = self._original["site_sameframe"]
        self._forward()
        self._active = False
        return True

    def close(self) -> None:
        """Restore the reused invisible site without closing the environment."""
        self.restore()
        self._closed = True
