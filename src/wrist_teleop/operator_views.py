"""Operator views for camera-only teleoperation: plan (top) and elevation (side).

The tri-view's two thumbnails are re-aimed as orthographic MuJoCo cameras
aligned with the operator's control basis, like the views of a technical
drawing:

* top (plan): looking straight down, image right = the operator's right,
  image up = "away" (hand held nearer the camera). With no perspective, the
  hand drawn exactly over the bowl *is* over the bowl, at any height.
* side (elevation): looking horizontally along the right axis, image up =
  world up, so the gap between the palm and the table / bowl reads directly.

``OperatorOverlay`` then draws only pixels on the composed image: a palm
marker in both views, the drop line and clearance in the side view, and the
webcam preview with its safe zone and depth gauge in the main view. Nothing
here changes the simulation state or the robot command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .camera_arm_control import _matrix_to_quat

RED = (255, 40, 40)
GREEN = (60, 220, 60)
YELLOW = (255, 210, 40)
BLUE = (80, 170, 255)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)


@dataclass(frozen=True)
class OrthoView:
    camera_name: str
    position: np.ndarray  # world
    x_axis: np.ndarray  # world direction of image right
    y_axis: np.ndarray  # world direction of image up
    extent_m: float  # visible height (MuJoCo orthographic ``fovy``)
    size_px: tuple[int, int]  # (width, height)

    @property
    def quat_wxyz(self) -> np.ndarray:
        z_axis = np.cross(self.x_axis, self.y_axis)  # MuJoCo cameras look along -z
        return _matrix_to_quat(np.column_stack((self.x_axis, self.y_axis, z_axis)))

    def project(self, point: Any) -> tuple[int, int]:
        width, height = self.size_px
        relative = np.asarray(point, dtype=np.float64) - self.position
        scale = height / self.extent_m
        return (int(round(width / 2 + float(relative @ self.x_axis) * scale)),
                int(round(height / 2 - float(relative @ self.y_axis) * scale)))


def horizontal_axes(world_from_control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Level 'right' and 'away' world directions of the operator basis."""

    right = np.array(world_from_control, dtype=np.float64)[:, 0].copy()
    right[2] = 0.0
    right /= np.linalg.norm(right)
    away = np.cross((0.0, 0.0, 1.0), right)  # right x away = up
    return right, away


def plan_and_elevation_views(
    points_xy: np.ndarray,
    z_range: tuple[float, float],
    world_from_control: np.ndarray,
    *,
    top_camera: str,
    top_size: tuple[int, int],
    side_camera: str,
    side_size: tuple[int, int],
    side_sign: float = -1.0,
    margin_m: float = 0.15,
    min_extent_m: float = 0.55,
) -> tuple[OrthoView, OrthoView]:
    """Views framing ``points_xy`` (objects, robot hand) and ``z_range``.

    ``side_sign`` -1 puts the side camera on the operator's left of the scene
    (image right = toward the main camera), +1 on the right (image right =
    away); pick the side that is not hidden behind tall fixtures.
    """

    right, away = horizontal_axes(world_from_control)
    up = np.array((0.0, 0.0, 1.0))
    points = np.asarray(points_xy, dtype=np.float64)
    r = points @ right[:2]
    a = points @ away[:2]
    r_lo, r_hi, a_lo, a_hi = r.min() - margin_m, r.max() + margin_m, a.min() - margin_m, a.max() + margin_m
    centre_r, centre_a = 0.5 * (r_lo + r_hi), 0.5 * (a_lo + a_hi)
    z_lo, z_hi = z_range
    top_aspect = top_size[0] / top_size[1]
    top_extent = max(a_hi - a_lo, (r_hi - r_lo) / top_aspect, min_extent_m)
    top = OrthoView(
        top_camera, centre_r * right + centre_a * away + up * (z_hi + 1.0),
        right.copy(), away.copy(), float(top_extent), tuple(top_size),
    )
    side_aspect = side_size[0] / side_size[1]
    side_extent = max(z_hi - z_lo, (a_hi - a_lo) / side_aspect, 0.6 * min_extent_m)
    look_from = float(np.sign(side_sign) or -1.0) * right  # camera +z points back toward it
    side = OrthoView(
        side_camera, centre_r * right + centre_a * away + look_from * 1.6 + up * 0.5 * (z_lo + z_hi),
        np.cross(up, look_from), up.copy(), float(side_extent), tuple(side_size),
    )
    return top, side


def apply_ortho_view(sim: Any, view: OrthoView) -> None:
    """Point a world-fixed MuJoCo camera as ``view`` (visual settings only)."""

    import mujoco

    model = getattr(sim.model, "_model", sim.model)
    camera = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view.camera_name))
    if camera < 0 or int(model.cam_bodyid[camera]) != 0:
        raise ValueError(f"need a world camera, got {view.camera_name!r}")
    model.cam_pos[camera] = view.position
    model.cam_quat[camera] = view.quat_wxyz
    model.cam_fovy[camera] = view.extent_m
    if hasattr(model, "cam_orthographic"):
        model.cam_orthographic[camera] = 1
    sim.forward()


def _cross(cv2: Any, frame: np.ndarray, centre: tuple[int, int], color, size: int = 12) -> None:
    x, y = centre
    cv2.line(frame, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), size // 2, color, 1, cv2.LINE_AA)


def _text(cv2: Any, frame: np.ndarray, text: str, origin: tuple[int, int], color=WHITE, scale: float = 0.55,
          thickness: int = 1) -> None:
    """Text on a darkened box, readable on the bright walls and the table."""

    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    box = frame[max(y - height - 4, 0):max(y + baseline + 2, 0), max(x - 4, 0):max(x + width + 4, 0)]
    box[:] = (box * 0.4).astype(np.uint8)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class OperatorOverlay:
    """Draws the operator aids onto the composed tri-view (RGB) image."""

    def __init__(self, top: OrthoView, side: OrthoView, *, depth_deadzone_m: float, edge_warning: float,
                 edge_margin: float = 0.01, depth_gauge_range_m: float = 0.12, inset_width_px: int = 400):
        self.top, self.side = top, side
        self.depth_deadzone_m, self.depth_gauge_range_m = float(depth_deadzone_m), float(depth_gauge_range_m)
        self.edge_warning, self.edge_margin = float(edge_warning), float(edge_margin)
        self.inset_width_px = int(inset_width_px)
        self.state: dict[str, Any] = {}

    def __call__(self, frame: np.ndarray, rects: dict[str, tuple[int, int, int, int]]) -> None:
        import cv2

        state = dict(self.state)
        palm = state.get("palm_world")
        hit = state.get("hit_world")
        for view, name, label in ((self.top, "wrist", "TOP (plan)   up = away"),
                                  (self.side, "side", "SIDE (elevation)")):
            x0, y0, width, height = rects[name]
            panel = frame[y0:y0 + height, x0:x0 + width]
            _text(cv2, panel, label, (8, 22), WHITE, 0.55)
            if palm is None:
                continue
            centre = view.project(palm)
            if view is self.side and hit is not None:
                ground = view.project(hit)
                cv2.line(panel, centre, ground, RED, 1, cv2.LINE_AA)
                cv2.line(panel, (ground[0] - 14, ground[1]), (ground[0] + 14, ground[1]), RED, 2, cv2.LINE_AA)
                clearance = state.get("clearance_m")
                if clearance is not None:
                    _text(cv2, panel, f"{100 * clearance:.1f} cm", (centre[0] + 14, (centre[1] + ground[1]) // 2),
                          YELLOW, 0.6, 2)
            if view is self.top and hit is not None:
                centre = view.project(hit)
            _cross(cv2, panel, centre, RED)
        self._draw_main(cv2, frame, rects["front"], state)

    def _draw_main(self, cv2: Any, frame: np.ndarray, rect: tuple[int, int, int, int], state: dict[str, Any]) -> None:
        x0, y0, width, height = rect
        panel = frame[y0:y0 + height, x0:x0 + width]
        for row, (line, color) in enumerate(state.get("lines", ())):
            _text(cv2, panel, line, (12, 52 + 26 * row), color, 0.62, 1)
        camera = state.get("camera_frame")
        if camera is None:
            return
        inset_w = self.inset_width_px
        inset_h = int(round(inset_w * camera.shape[0] / camera.shape[1]))
        inset = cv2.resize(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB), (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        # Safe zone: wrist and knuckles should stay inside the inner box.
        edge = state.get("edge_distance")
        lost_at_border = "border" in str(state.get("pose_error") or "")
        if lost_at_border or (edge is not None and edge < 2.0 * self.edge_margin):
            zone_color = RED
        elif edge is not None and edge < self.edge_warning:
            zone_color = YELLOW
        else:
            zone_color = GREEN
        mx, my = int(self.edge_warning * inset_w), int(self.edge_warning * inset_h)
        cv2.rectangle(inset, (mx, my), (inset_w - mx, inset_h - my), zone_color, 2)
        badge = state.get("badge")
        if badge:
            _text(cv2, inset, badge[0], (12, 34), badge[1], 0.9, 2)
        gauge_w, gap = 34, 8
        left, top = 12, height - inset_h - 30  # room for the gauge's lower label
        panel[top:top + inset_h, left:left + inset_w] = inset
        cv2.rectangle(panel, (left - 1, top - 1), (left + inset_w, top + inset_h), WHITE, 1)
        self._draw_depth_gauge(cv2, panel, (left + inset_w + gap, top, gauge_w, inset_h), state)

    def _draw_depth_gauge(self, cv2: Any, panel: np.ndarray, rect: tuple[int, int, int, int],
                          state: dict[str, Any]) -> None:
        x, y, width, height = rect
        cv2.rectangle(panel, (x, y), (x + width, y + height), (30, 30, 30), -1)
        span = self.depth_gauge_range_m
        centre_y = y + height // 2

        def to_y(offset: float) -> int:
            return int(round(centre_y - float(np.clip(offset / span, -1.0, 1.0)) * (height // 2 - 4)))

        cv2.rectangle(panel, (x, to_y(self.depth_deadzone_m)), (x + width, to_y(-self.depth_deadzone_m)), (70, 70, 70), -1)
        cv2.line(panel, (x, centre_y), (x + width, centre_y), GREY, 1)
        offset = state.get("depth_offset_m")
        if offset is not None:
            active = abs(offset) > self.depth_deadzone_m
            marker = to_y(offset)
            cv2.rectangle(panel, (x + 3, marker - 4), (x + width - 3, marker + 4), YELLOW if active else GREEN, -1)
        _text(cv2, panel, "away", (x - 2, y - 6), WHITE, 0.45)
        _text(cv2, panel, "back", (x - 2, y + height + 16), WHITE, 0.45)
