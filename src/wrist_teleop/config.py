"""Validated, centralized settings for the SpaceMouse wrist control chain."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hand_gestures import DEFAULT_HAND_GESTURE_CONFIG, HandGestureConfig
from .grasp_telemetry import GraspTelemetryConfig


def _finite_number(name: str, value: Any, *, minimum: float = 0.0,
                   inclusive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result) or (result < minimum if inclusive else result <= minimum):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return result


def _sequence(name: str, value: Any, length: int) -> tuple:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must contain exactly {length} values")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly {length} values") from exc
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _boolean(name: str, value: Any) -> bool:
    """Require a real JSON-style boolean instead of silently coercing values."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty text value")
    return value


@dataclass(frozen=True)
class TriViewConfig:
    """Settings for the single-window, synchronized three-camera display.

    Camera names are MuJoCo camera names.  ``front`` is the large primary
    image; ``wrist`` and ``side`` are thumbnails.  The compositor renders all
    three from one already-stepped simulation state and never sleeps, so the
    configured rate only limits rendering work rather than the control loop.
    """

    enabled: bool = True
    front_camera_name: str = "frontview"
    wrist_camera_name: str = "robot0_eye_in_hand"
    side_camera_name: str = "sideview"
    # ``image_*`` is the native front-panel render size, not a low-resolution
    # source that will later be enlarged by OpenCV.  Together with the
    # thumbnail layout below this produces a 1920 x 1080 composite.
    image_width: int = 1530
    image_height: int = 1080
    render_hz: float = 20.0
    layout: str = "main_left"
    thumbnail_scale: float = 380.0 / 1530.0
    gutter_px: int = 10
    # Saved composites always keep their native 1920 x 1080 pixels.  The
    # optional display size is only a practical fit-to-desktop request for
    # HighGUI; set both to null to request native-sized display too.
    display_width: int | None = 1600
    display_height: int | None = 900
    window_name: str = "Robosuite synchronized tri-view"
    show_window: bool = True
    # The primary view is behind the Panda's left shoulder, roughly 35° from
    # its rear axis.  It shows the wrist, hand, cube, table, and TCP laser in
    # one view.  Wrist is body-attached, so only its field of view is
    # overridden.
    front_position_m: tuple[float, float, float] = (-1.55, -1.10, 1.52)
    front_quat_wxyz: tuple[float, float, float, float] = (
        0.7239870608246975, 0.5149914938413807, -0.2660206107280316, -0.3739779829045074
    )
    front_fovy_deg: float = 45.0
    side_position_m: tuple[float, float, float] = (-0.05651774593317116, 1.2761224129427358, 1.4879572214102434)
    side_quat_wxyz: tuple[float, float, float, float] = (
        0.009905065491771751, 0.006877963156909582, 0.5912228352893879, 0.806418094001364
    )
    side_fovy_deg: float = 45.0
    wrist_fovy_deg: float = 75.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean("tri_view.enabled", self.enabled))
        object.__setattr__(self, "show_window", _boolean("tri_view.show_window", self.show_window))
        names = tuple(
            _text(f"tri_view.{field}", getattr(self, field))
            for field in ("front_camera_name", "wrist_camera_name", "side_camera_name")
        )
        if len(set(names)) != 3:
            raise ValueError("tri_view camera names must be distinct")
        for field, value in zip(("front_camera_name", "wrist_camera_name", "side_camera_name"), names):
            object.__setattr__(self, field, value)
        for field in ("image_width", "image_height"):
            value = getattr(self, field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 16:
                raise ValueError(f"tri_view.{field} must be an integer of at least 16")
            object.__setattr__(self, field, int(value))
        object.__setattr__(self, "render_hz", _finite_number("tri_view.render_hz", self.render_hz))
        if self.layout not in {"main_left", "main_top"}:
            raise ValueError("tri_view.layout must be 'main_left' or 'main_top'")
        scale = _finite_number("tri_view.thumbnail_scale", self.thumbnail_scale)
        if scale > 1.0:
            raise ValueError("tri_view.thumbnail_scale must be in (0, 1]")
        object.__setattr__(self, "thumbnail_scale", scale)
        gutter = self.gutter_px
        if isinstance(gutter, (bool, np.bool_)) or not isinstance(gutter, Integral) or int(gutter) < 0:
            raise ValueError("tri_view.gutter_px must be a nonnegative integer")
        gutter = int(gutter)
        if gutter >= min(self.image_width, self.image_height) - 16:
            raise ValueError("tri_view.gutter_px leaves no usable thumbnail area")
        object.__setattr__(self, "gutter_px", gutter)
        display_values = (self.display_width, self.display_height)
        if any(value is None for value in display_values):
            if any(value is not None for value in display_values):
                raise ValueError("tri_view.display_width and display_height must both be integers or null")
        else:
            for field, value in zip(("display_width", "display_height"), display_values):
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 16:
                    raise ValueError(f"tri_view.{field} must be an integer of at least 16 or null")
            object.__setattr__(self, "display_width", int(self.display_width))
            object.__setattr__(self, "display_height", int(self.display_height))
        # Ensure derived panels have real render sizes.  They are rendered at
        # these exact dimensions, so composition never enlarges a tiny image.
        if self.layout == "main_left":
            thumb_width = max(16, int(round(self.image_width * scale)))
            upper = (self.image_height - gutter) // 2
            lower = self.image_height - gutter - upper
            if thumb_width < 16 or min(upper, lower) < 16:
                raise ValueError("tri_view layout leaves no usable thumbnail area")
        else:
            thumb_height = max(16, int(round(self.image_height * scale)))
            left = (self.image_width - gutter) // 2
            right = self.image_width - gutter - left
            if thumb_height < 16 or min(left, right) < 16:
                raise ValueError("tri_view layout leaves no usable thumbnail area")
        object.__setattr__(self, "window_name", _text("tri_view.window_name", self.window_name))
        for field in ("front_position_m", "side_position_m"):
            vector = _sequence(f"tri_view.{field}", getattr(self, field), 3)
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in vector):
                raise ValueError(f"tri_view.{field} must contain three finite numbers")
            parsed = tuple(float(value) for value in vector)
            if not np.all(np.isfinite(parsed)):
                raise ValueError(f"tri_view.{field} must contain three finite numbers")
            object.__setattr__(self, field, parsed)
        for field in ("front_quat_wxyz", "side_quat_wxyz"):
            vector = _sequence(f"tri_view.{field}", getattr(self, field), 4)
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in vector):
                raise ValueError(f"tri_view.{field} must contain four finite numbers")
            quaternion = np.asarray(vector, dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if not np.all(np.isfinite(quaternion)) or norm <= 1.0e-12:
                raise ValueError(f"tri_view.{field} must be a finite nonzero quaternion")
            object.__setattr__(self, field, tuple(float(value / norm) for value in quaternion))
        for field in ("front_fovy_deg", "side_fovy_deg", "wrist_fovy_deg"):
            value = _finite_number(f"tri_view.{field}", getattr(self, field))
            if value >= 180.0:
                raise ValueError(f"tri_view.{field} must be in (0, 180)")
            object.__setattr__(self, field, value)

    @property
    def camera_names(self) -> tuple[str, str, str]:
        """Configured camera names in displayed front / wrist / side order."""
        return self.front_camera_name, self.wrist_camera_name, self.side_camera_name

    @property
    def fixed_camera_overrides(self) -> dict[str, dict[str, tuple[float, ...] | float | None]]:
        """Camera pose / optics overrides in the same front-wrist-side order.

        The wrist camera remains body-attached; a world-space pose is therefore
        intentionally ``None`` while its configured field of view still applies.
        """
        return {
            "front": {
                "position_m": self.front_position_m,
                "quat_wxyz": self.front_quat_wxyz,
                "fovy_deg": self.front_fovy_deg,
            },
            "wrist": {"position_m": None, "quat_wxyz": None, "fovy_deg": self.wrist_fovy_deg},
            "side": {
                "position_m": self.side_position_m,
                "quat_wxyz": self.side_quat_wxyz,
                "fovy_deg": self.side_fovy_deg,
            },
        }

    @property
    def panel_sizes(self) -> dict[str, tuple[int, int]]:
        """Native ``(width, height)`` render size for each displayed panel.

        The compositor asks MuJoCo for these final panel dimensions directly.
        This avoids the former 640 x 480 source followed by an upsampled UI.
        """
        width, height, gutter = self.image_width, self.image_height, self.gutter_px
        if self.layout == "main_left":
            thumb_width = max(16, int(round(width * self.thumbnail_scale)))
            upper = (height - gutter) // 2
            lower = height - gutter - upper
            return {"front": (width, height), "wrist": (thumb_width, upper), "side": (thumb_width, lower)}
        thumb_height = max(16, int(round(height * self.thumbnail_scale)))
        left = (width - gutter) // 2
        right = width - gutter - left
        return {"front": (width, height), "wrist": (left, thumb_height), "side": (right, thumb_height)}

    @property
    def composite_size(self) -> tuple[int, int]:
        """Final ``(width, height)`` of the saved RGB composite."""
        if self.layout == "main_left":
            thumb_width, _ = self.panel_sizes["wrist"]
            return self.image_width + self.gutter_px + thumb_width, self.image_height
        _, thumb_height = self.panel_sizes["wrist"]
        return self.image_width, self.image_height + self.gutter_px + thumb_height


@dataclass(frozen=True)
class LaserConfig:
    """Visual-only TCP laser settings shared by all supported teleoperation UIs.

    ``start_site`` provides the true TCP origin and ``visual_site`` names the
    existing transparent MuJoCo cylinder site used only for rendering.  The
    direction normally is expressed in the start site's local coordinate
    frame.  A caller can instead request a fixed world-frame vector; Panda +
    Allegro uses that mode for its palm-to-table targeting aid.  View names
    are the compositor's logical ``front``, ``wrist``, and ``side`` panels,
    so camera names can be changed in :class:`TriViewConfig` without
    duplicating this list.
    """

    enabled: bool = True
    color_rgba: tuple[float, float, float, float] = (1.0, 0.05, 0.05, 0.90)
    width_m: float = 0.003
    max_length_m: float = 1.20
    start_site: str = "grip_site"
    visual_site: str = "grip_site_cylinder"
    local_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    direction_frame: str = "site"
    world_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    visible_views: tuple[str, ...] = ("front", "wrist", "side")
    hide_on_no_hit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _boolean("laser.enabled", self.enabled))
        object.__setattr__(self, "hide_on_no_hit", _boolean("laser.hide_on_no_hit", self.hide_on_no_hit))
        color = _sequence("laser.color_rgba", self.color_rgba, 4)
        parsed_color: list[float] = []
        for index, value in enumerate(color):
            value = _finite_number(f"laser.color_rgba[{index}]", value, minimum=0.0, inclusive=True)
            if value > 1.0:
                raise ValueError("laser.color_rgba values must be in [0, 1]")
            parsed_color.append(value)
        object.__setattr__(self, "color_rgba", tuple(parsed_color))
        object.__setattr__(self, "width_m", _finite_number("laser.width_m", self.width_m))
        object.__setattr__(self, "max_length_m", _finite_number("laser.max_length_m", self.max_length_m))
        object.__setattr__(self, "start_site", _text("laser.start_site", self.start_site))
        object.__setattr__(self, "visual_site", _text("laser.visual_site", self.visual_site))
        if self.start_site == self.visual_site:
            raise ValueError("laser.start_site and laser.visual_site must be different")
        direction = _sequence("laser.local_direction", self.local_direction, 3)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in direction):
            raise ValueError("laser.local_direction must contain three finite numbers")
        direction_array = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction_array))
        if not np.all(np.isfinite(direction_array)) or norm <= 1.0e-12:
            raise ValueError("laser.local_direction must have a finite nonzero length")
        object.__setattr__(self, "local_direction", tuple(float(value / norm) for value in direction_array))
        if self.direction_frame not in {"site", "world"}:
            raise ValueError("laser.direction_frame must be 'site' or 'world'")
        world_direction = _sequence("laser.world_direction", self.world_direction, 3)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in world_direction):
            raise ValueError("laser.world_direction must contain three finite numbers")
        world_array = np.asarray(world_direction, dtype=np.float64)
        world_norm = float(np.linalg.norm(world_array))
        if not np.all(np.isfinite(world_array)) or world_norm <= 1.0e-12:
            raise ValueError("laser.world_direction must have a finite nonzero length")
        object.__setattr__(self, "world_direction", tuple(float(value / world_norm) for value in world_array))
        if isinstance(self.visible_views, (str, bytes, Mapping)):
            raise ValueError("laser.visible_views must be a sequence of front, wrist, and side")
        try:
            views = tuple(self.visible_views)
        except TypeError as exc:
            raise ValueError("laser.visible_views must be a sequence of front, wrist, and side") from exc
        allowed = {"front", "wrist", "side"}
        if any(not isinstance(view, str) or view not in allowed for view in views) or len(set(views)) != len(views):
            raise ValueError("laser.visible_views must contain unique front, wrist, and/or side names")
        object.__setattr__(self, "visible_views", views)


DEFAULT_TRI_VIEW_CONFIG = TriViewConfig()
DEFAULT_LASER_CONFIG = LaserConfig()


@dataclass(frozen=True)
class DexHandConfig:
    """Camera-frame policy for the optional official dex hand solver.

    ``stale_timeout_s`` is the bounded grace period during which the last
    valid dex target may be held after MediaPipe loses a hand.  On expiry the
    gesture layer switches to its configured safe-open target; no stale target
    is reused indefinitely.
    """

    stale_timeout_s: float = 0.5
    expected_handedness: str = "Right"
    input_is_mirrored: bool = False
    runtime: str = "auto"
    sidecar_python: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stale_timeout_s", _finite_number(
            "dex_hand.stale_timeout_s", self.stale_timeout_s
        ))
        handedness = _text("dex_hand.expected_handedness", self.expected_handedness).lower()
        if handedness not in {"left", "right"}:
            raise ValueError("dex_hand.expected_handedness must be 'Left' or 'Right'")
        object.__setattr__(self, "expected_handedness", handedness.title())
        object.__setattr__(self, "input_is_mirrored", _boolean(
            "dex_hand.input_is_mirrored", self.input_is_mirrored
        ))
        runtime = _text("dex_hand.runtime", self.runtime).lower()
        if runtime not in {"auto", "native", "sidecar"}:
            raise ValueError("dex_hand.runtime must be 'auto', 'native', or 'sidecar'")
        object.__setattr__(self, "runtime", runtime)
        if self.sidecar_python is not None:
            object.__setattr__(
                self,
                "sidecar_python",
                _text("dex_hand.sidecar_python", self.sidecar_python),
            )


DEFAULT_DEX_HAND_CONFIG = DexHandConfig()
DEFAULT_GRASP_TELEMETRY_CONFIG = GraspTelemetryConfig()


@dataclass(frozen=True)
class WristConfig:
    """All per-step wrist and button-gesture settings in one validated object.

    All wrist deltas are metres / radians. ``hand_gestures`` is the only
    source of the SpaceMouse button indices, gesture curls, and transition
    duration used by the integrated Panda + Allegro entry point.

    HID axes enter as [Tx, Ty, Tz, Rx, Ry, Rz]. ``axis_order[j]`` selects
    the input axis for mapped device axis j, then ``axis_inversion[j]`` is
    applied. Permutations must stay within each translation/rotation triad.
    ``control_frame`` selects how those six mapped device axes become MuJoCo
    world-frame OSC increments.  ``world`` uses ``device_to_world`` directly.
    ``camera`` derives a right / depth / world-up basis from the primary
    tri-view camera's actual world extrinsics; this keeps left / right motion
    visually consistent when the shoulder camera moves.  Axis-angle vectors
    describe extrinsic world-axis increments in either case.

    Deadzones are fractions of raw_ranges, with continuous linear rescaling
    outside the deadzone. EMA alpha is the new sample weight; None disables
    smoothing. All settings are immutable after validation.
    """

    raw_ranges: tuple[float, ...] = (350.0,) * 6
    axis_order: tuple[int, ...] = (1, 0, 2, 3, 4, 5)
    axis_inversion: tuple[int, ...] = (1, 1, -1, 1, 1, 1)
    device_to_world: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    control_frame: str = "camera"
    coordinate_frame: str = "world"
    translation_deadzone: float = 0.08
    rotation_deadzone: float = 0.08
    # Entering an input band uses the existing deadzone. Releasing it uses a
    # smaller threshold so near-centre HID noise cannot repeatedly re-arm a
    # filtered command. Both values are fractions of ``raw_ranges``.
    translation_release_deadzone: float = 0.06
    rotation_release_deadzone: float = 0.06
    # Zero bias is learned only once after opening / reset while the device is
    # held in its neutral band. It is never adapted during normal motion.
    zero_bias_calibration_frames: int = 8
    neutral_lock_frames: int = 3
    translation_scale_m: float = 0.003
    rotation_scale_rad: float = 0.025
    max_translation_delta_m: float = 0.004
    max_rotation_delta_rad: float = 0.04
    smoothing_alpha: float | None = 0.35
    control_hz: float = 20.0
    stale_timeout_s: float = 0.25
    discovery_interval_s: float = 0.5
    hand_gestures: HandGestureConfig = DEFAULT_HAND_GESTURE_CONFIG
    tri_view: TriViewConfig = DEFAULT_TRI_VIEW_CONFIG
    laser: LaserConfig = DEFAULT_LASER_CONFIG
    dex_hand: DexHandConfig = DEFAULT_DEX_HAND_CONFIG
    grasp_telemetry: GraspTelemetryConfig = DEFAULT_GRASP_TELEMETRY_CONFIG
    neutral_startup_s: float = 0.3
    vendor_id: int | None = None
    product_id: int | None = None
    output_dir: str = "outputs/spacemouse_wrist"

    def __post_init__(self) -> None:
        ranges = _sequence("raw_ranges", self.raw_ranges, 6)
        object.__setattr__(self, "raw_ranges", tuple(
            _finite_number(f"raw_ranges[{i}]", value) for i, value in enumerate(ranges)
        ))
        order = _sequence("axis_order", self.axis_order, 6)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in order):
            raise ValueError("axis_order must contain integer axis indices")
        if set(order[:3]) != {0, 1, 2} or set(order[3:]) != {3, 4, 5}:
            raise ValueError("axis_order must permute translation and rotation axes separately")
        object.__setattr__(self, "axis_order", tuple(int(x) for x in order))
        inversion = _sequence("axis_inversion", self.axis_inversion, 6)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real)
               or x not in (-1, 1) for x in inversion):
            raise ValueError("axis_inversion must contain only -1 or +1")
        object.__setattr__(self, "axis_inversion", tuple(int(x) for x in inversion))

        rows = _sequence("device_to_world", self.device_to_world, 3)
        rotation_rows = tuple(_sequence("device_to_world row", row, 3) for row in rows)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real)
               for row in rotation_rows for x in row):
            raise ValueError("device_to_world must be a finite proper 3x3 rotation")
        rotation = np.asarray(rotation_rows, dtype=np.float64)
        if (not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0)):
            raise ValueError("device_to_world must be a finite proper 3x3 rotation")
        object.__setattr__(self, "device_to_world", tuple(tuple(float(x) for x in row)
                                                         for row in rotation))
        if self.coordinate_frame != "world":
            raise ValueError("coordinate_frame must be 'world'")
        if not isinstance(self.control_frame, str) or self.control_frame not in {"world", "camera"}:
            raise ValueError("control_frame must be 'world' or 'camera'")
        for name in ("translation_deadzone", "rotation_deadzone"):
            value = _finite_number(name, getattr(self, name), inclusive=True)
            if value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        for release_name, enter_name in (
            ("translation_release_deadzone", "translation_deadzone"),
            ("rotation_release_deadzone", "rotation_deadzone"),
        ):
            value = _finite_number(release_name, getattr(self, release_name), inclusive=True)
            if value > getattr(self, enter_name):
                raise ValueError(f"{release_name} must be in [0, {enter_name}]")
            object.__setattr__(self, release_name, value)
        for name in ("zero_bias_calibration_frames", "neutral_lock_frames"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 1 <= int(value) <= 120:
                raise ValueError(f"{name} must be an integer in [1, 120]")
            object.__setattr__(self, name, int(value))
        for name in ("translation_scale_m", "rotation_scale_rad", "max_translation_delta_m",
                     "max_rotation_delta_rad", "control_hz", "stale_timeout_s",
                     "discovery_interval_s", "neutral_startup_s"):
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))
        if self.smoothing_alpha is not None:
            alpha = _finite_number("smoothing_alpha", self.smoothing_alpha)
            if alpha > 1.0:
                raise ValueError("smoothing_alpha must be in (0, 1] or null")
            object.__setattr__(self, "smoothing_alpha", alpha)
        gestures = self.hand_gestures
        if isinstance(gestures, Mapping):
            gestures = HandGestureConfig.from_dict(gestures)
        if not isinstance(gestures, HandGestureConfig):
            raise ValueError("hand_gestures must be a HandGestureConfig or JSON object")
        object.__setattr__(self, "hand_gestures", gestures)
        tri_view = self.tri_view
        if isinstance(tri_view, Mapping):
            tri_view = TriViewConfig(**dict(tri_view))
        if not isinstance(tri_view, TriViewConfig):
            raise ValueError("tri_view must be a TriViewConfig or JSON object")
        object.__setattr__(self, "tri_view", tri_view)
        laser = self.laser
        if isinstance(laser, Mapping):
            laser = LaserConfig(**dict(laser))
        if not isinstance(laser, LaserConfig):
            raise ValueError("laser must be a LaserConfig or JSON object")
        object.__setattr__(self, "laser", laser)
        dex_hand = self.dex_hand
        if isinstance(dex_hand, Mapping):
            dex_hand = DexHandConfig(**dict(dex_hand))
        if not isinstance(dex_hand, DexHandConfig):
            raise ValueError("dex_hand must be a DexHandConfig or JSON object")
        object.__setattr__(self, "dex_hand", dex_hand)
        grasp_telemetry = self.grasp_telemetry
        if isinstance(grasp_telemetry, Mapping):
            grasp_telemetry = GraspTelemetryConfig(**dict(grasp_telemetry))
        if not isinstance(grasp_telemetry, GraspTelemetryConfig):
            raise ValueError("grasp_telemetry must be a GraspTelemetryConfig or JSON object")
        object.__setattr__(self, "grasp_telemetry", grasp_telemetry)

        for name, maximum in (("vendor_id", 65535), ("product_id", 65535)):
            value = getattr(self, name)
            if value is not None:
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                        or not 0 <= value <= maximum):
                    raise ValueError(f"{name} must be an integer in [0, {maximum}] or null")
                object.__setattr__(self, name, int(value))
        if (not isinstance(self.output_dir, str) or not self.output_dir.strip()
                or "\x00" in self.output_dir):
            raise ValueError("output_dir must be a nonempty path string")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> WristConfig:
        if not isinstance(values, Mapping):
            raise ValueError("configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration settings: {sorted(unknown, key=str)}")
        return cls(**dict(values))

    @classmethod
    def load(cls, path: str | Path | None = None) -> WristConfig:
        """Load JSON overrides, or return validated defaults when path is None."""
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            values = json.load(stream)
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary suitable for JSON metadata logging."""
        return asdict(self)
