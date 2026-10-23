"""Synthetic MediaPipe-style right-hand detections for camera-control tests.

A rigid 21-landmark right hand is placed in the raw camera frame (x right
in the raw image, y down, z away from the camera), projected with a pinhole
camera, then flipped like ``CameraHandWorker``'s mirrored selfie frame. The
outputs therefore match what the real detector delivers, so the estimator's
direction conventions can be checked without a webcam.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from .camera_arm_control import CAMERA_TO_CONTROL, palm_frame, rotvec_to_matrix

# Physical right hand, palm toward the camera, fingers up (raw camera frame,
# metres, relative to the wrist). The thumb is on the raw-image right.
RIGHT_HAND_POINTS = np.array((
    (0.000, 0.000, 0.000),
    (0.025, -0.025, -0.005), (0.045, -0.045, -0.010), (0.060, -0.065, -0.015), (0.070, -0.085, -0.020),
    (0.025, -0.090, 0.000), (0.027, -0.130, -0.005), (0.028, -0.155, -0.008), (0.029, -0.175, -0.010),
    (0.005, -0.095, 0.000), (0.005, -0.140, -0.005), (0.005, -0.165, -0.008), (0.005, -0.185, -0.010),
    (-0.015, -0.090, 0.000), (-0.016, -0.130, -0.005), (-0.017, -0.152, -0.008), (-0.018, -0.170, -0.010),
    (-0.033, -0.080, 0.000), (-0.036, -0.110, -0.004), (-0.038, -0.127, -0.006), (-0.040, -0.142, -0.008),
))


class SyntheticDetection:
    def __init__(self, image: np.ndarray, world: np.ndarray, frame_shape: tuple[int, int]):
        self.image_landmarks = image
        self.world_landmarks = world
        self.frame_shape = frame_shape


def synthesize(
    position_m: np.ndarray,
    rotation_camera: np.ndarray,
    *,
    focal_width_normalized: float = 0.78,
    frame_shape: tuple[int, int] = (480, 640),
    mirrored: bool = True,
    image_noise: float = 0.0,
    world_noise_m: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> SyntheticDetection:
    """One detection of the hand with its wrist at ``position_m`` (raw camera)."""

    rng = rng or np.random.default_rng()
    points = (np.asarray(rotation_camera) @ RIGHT_HAND_POINTS.T).T + np.asarray(position_m)
    height, width = frame_shape
    aspect = height / width
    u = focal_width_normalized * points[:, 0] / points[:, 2] + 0.5
    v = focal_width_normalized * points[:, 1] / points[:, 2] + 0.5 * aspect
    image = np.column_stack((1.0 - u if mirrored else u, v / aspect, np.zeros(21)))
    world = points - points.mean(axis=0)
    if mirrored:
        world[:, 0] = -world[:, 0]
    if image_noise:
        image[:, :2] += rng.normal(0.0, image_noise, size=(21, 2))
    if world_noise_m:
        world += rng.normal(0.0, world_noise_m, size=world.shape)
    return SyntheticDetection(image, world, frame_shape)


def control_rotation_to_camera(rotvec_control: np.ndarray) -> np.ndarray:
    """A rotation given in the operator basis [right, away, up], in camera axes."""

    return CAMERA_TO_CONTROL.T @ rotvec_to_matrix(rotvec_control) @ CAMERA_TO_CONTROL


def local_rotation_to_camera(rotvec_local: np.ndarray) -> np.ndarray:
    """A rotation in the hand's own palm frame (x fingers, z back of the hand)."""

    rest = palm_frame(RIGHT_HAND_POINTS)
    return rest @ rotvec_to_matrix(rotvec_local) @ rest.T


def control_offset_to_camera(offset_control: np.ndarray) -> np.ndarray:
    return CAMERA_TO_CONTROL.T @ np.asarray(offset_control, dtype=np.float64)


def _ramp(t: float, t0: float, t1: float) -> float:
    return float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


def scripted_pose(t: float, start=np.array((0.0, 0.03, 0.50))) -> tuple[np.ndarray, np.ndarray]:
    """Smoke script (arm anchors after ~1.5 s: palm template warm-up + stable frames).

    2.5-4.0 s right 10 cm; 4.0-4.5 s 6 cm toward the camera, held to 5.5 s
    (depth rate control: the robot moves away), back to neutral by 6.0 s;
    6.0-7.2 s up 6 cm; 7.2-8.6 s knob turn 25 deg (robot yaw); 9.0-9.6 s the
    hand moves back left 10 cm while the left-hand clutch is shown
    (``scripted_clutch``), so the robot must not follow.
    """

    offset = np.array((
        0.10 * _ramp(t, 2.5, 4.0) - 0.10 * _ramp(t, 9.0, 9.6),
        0.06 * _ramp(t, 4.0, 4.5) - 0.06 * _ramp(t, 5.5, 6.0),
        0.06 * _ramp(t, 6.0, 7.2),
    ))
    yaw = np.radians(25.0) * _ramp(t, 7.2, 8.6)
    position = np.asarray(start) + control_offset_to_camera(offset)
    rotation = local_rotation_to_camera(np.array((0.0, 0.0, yaw)))
    return position, rotation


def scripted_clutch(t: float) -> bool:
    """Left hand shown (clutch) during the smoke script's reposition."""

    return 8.8 <= t <= 9.9


class SyntheticHandWorker:
    """Stand-in for ``CameraHandWorker``: feeds scripted detections at 30 Hz."""

    def __init__(self, adapter: Any, script: Callable[[float], tuple[np.ndarray, np.ndarray]] = scripted_pose,
                 *, clutch_script: Optional[Callable[[float], bool]] = scripted_clutch,
                 rate_hz: float = 30.0, image_noise: float = 0.0015, world_noise_m: float = 0.002,
                 mirrored: bool = True, focal_width_normalized: float = 0.78):
        self.adapter, self.script, self.rate_hz = adapter, script, float(rate_hz)
        self.clutch_script = clutch_script
        self.image_noise, self.world_noise_m, self.mirrored = image_noise, world_noise_m, mirrored
        self.focal = focal_width_normalized
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.frames = 0
        self._latest: Optional[tuple[np.ndarray, tuple[int, int], bool, float]] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="synthetic-hand", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        rng = np.random.default_rng(7)
        started = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            position, rotation = self.script(now - started)
            detection = synthesize(position, rotation, focal_width_normalized=self.focal, mirrored=self.mirrored,
                                   image_noise=self.image_noise, world_noise_m=self.world_noise_m, rng=rng)
            clutch = bool(self.clutch_script and self.clutch_script(now - started))
            self._latest = (detection.image_landmarks.copy(), detection.frame_shape, clutch, now)
            try:
                self.adapter.update_hand_frame(detection.image_landmarks, detection.world_landmarks, "Right",
                                               now, detection.frame_shape,
                                               detected_hands=("Right", "Left") if clutch else ("Right",))
                self.frames += 1
            except Exception as exc:  # surface in the run log, never kill the control loop
                self.error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 / self.rate_hz)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def status(self) -> dict[str, Any]:
        return {"synthetic": True, "frames": self.frames, "error": self.error}

    def latest_frame(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        """A dark stand-in for the webcam preview with the synthetic landmarks (BGR)."""

        latest = self._latest
        if latest is None:
            return None, None
        image, (height, width), clutch, timestamp = latest
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        points = np.clip(image[:, :2] * (width, height), 0, (width - 1, height - 1)).astype(int)
        for x, y in points:
            frame[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4] = (60, 220, 60)
        if clutch:  # a block standing in for the left hand on the operator's left (image left)
            frame[height // 3:2 * height // 3, width // 12:width // 4] = (200, 160, 60)
        return frame, timestamp
