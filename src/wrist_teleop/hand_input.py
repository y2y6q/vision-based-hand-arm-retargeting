"""Camera-only rule-based Allegro hand input, extracted from the frozen demo.

Source behavior: ``scripts/run_camera_panda_allegro_teleop.py`` lines 228--520
(MediaPipe landmarks -> four curls -> sixteen Allegro targets).  This module
intentionally excludes camera wrist pose estimation, Panda IK, PyBullet, and
all SpaceMouse input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np


DEFAULT_ALLEGRO_JOINT_NAMES = tuple(f"joint_{index}.0" for index in range(16))
DEFAULT_ALLEGRO_LOWER_LIMITS = np.array(
    [-0.470, -0.196, -0.174, -0.227] * 3 + [0.263, -0.105, -0.189, -0.162], dtype=np.float64
)
DEFAULT_ALLEGRO_UPPER_LIMITS = np.array(
    [0.470, 1.610, 1.709, 1.618] * 3 + [1.396, 1.163, 1.644, 1.719], dtype=np.float64
)
NORMAL_FINGER_WEIGHTS = np.array([0.12, 0.80, 0.85, 0.85], dtype=np.float64)
THUMB_WEIGHTS = np.array([0.35, 0.65, 0.75, 0.80], dtype=np.float64)


@dataclass(frozen=True)
class HandCommand:
    """Latest camera hand command in Allegro model joint order."""

    joint_targets: np.ndarray
    valid: bool
    timestamp_s: float
    tracking_state: str
    calibration_state: str
    curls: np.ndarray
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _point(landmark: Any) -> np.ndarray:
    if hasattr(landmark, "x"):
        point = (landmark.x, landmark.y, landmark.z)
    else:
        point = landmark
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("each hand landmark must be three finite coordinates")
    return point


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-6 or second_norm < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def estimate_finger_curl(landmarks: Sequence[Any], ids: Sequence[int]) -> float:
    """Legacy angle-plus-distance curl estimator; 0=open and 1=closed."""
    mcp, pip, _dip, tip = (_point(landmarks[index]) for index in ids)
    angle_curl = float(np.clip((165.0 - _angle_degrees(mcp - pip, tip - pip)) / 95.0, 0.0, 1.0))
    wrist = _point(landmarks[0])
    distance_ratio = float(np.linalg.norm(tip - wrist) / (np.linalg.norm(mcp - wrist) + 1e-6))
    distance_curl = 1.0 - float(np.clip((distance_ratio - 1.1) / 1.1, 0.0, 1.0))
    return float(np.clip(0.75 * angle_curl + 0.25 * distance_curl, 0.0, 1.0))


def estimate_all_finger_curls(landmarks: Sequence[Any]) -> np.ndarray:
    if len(landmarks) != 21:
        raise ValueError("MediaPipe hand input must contain exactly 21 landmarks")
    index = estimate_finger_curl(landmarks, (5, 6, 7, 8))
    middle = estimate_finger_curl(landmarks, (9, 10, 11, 12))
    ring = estimate_finger_curl(landmarks, (13, 14, 15, 16))
    pinky = estimate_finger_curl(landmarks, (17, 18, 19, 20))
    thumb = estimate_finger_curl(landmarks, (1, 2, 3, 4))
    return np.array([index, middle, 0.70 * ring + 0.30 * pinky, thumb], dtype=np.float64)


def curls_to_allegro_targets(curls: Sequence[float], lower_limits: Sequence[float], upper_limits: Sequence[float]) -> np.ndarray:
    """Apply the exact legacy 4-curl / 16-target weighting with model limits."""
    curls = np.asarray(curls, dtype=np.float64)
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    if curls.shape != (4,) or lower.shape != (16,) or upper.shape != (16,):
        raise ValueError("expected curls (4,) and lower/upper limits (16,)")
    if not np.all(np.isfinite(curls)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(upper > lower):
        raise ValueError("curls and joint limits must be finite, ordered values")
    curls = np.clip(curls, 0.0, 1.0)
    opened = np.clip(np.zeros(16), lower, upper)
    targets = opened.copy()
    for finger in range(3):
        segment = slice(finger * 4, finger * 4 + 4)
        targets[segment] = opened[segment] + curls[finger] * NORMAL_FINGER_WEIGHTS * (upper[segment] - opened[segment])
    targets[12:16] = opened[12:16] + curls[3] * THUMB_WEIGHTS * (upper[12:16] - opened[12:16])
    return np.clip(targets, lower, upper)


class HandInputAdapter:
    """Thread-safe latest-value hand target source with legacy hold-on-loss safety."""

    def __init__(
        self,
        lower_limits: Sequence[float] = DEFAULT_ALLEGRO_LOWER_LIMITS,
        upper_limits: Sequence[float] = DEFAULT_ALLEGRO_UPPER_LIMITS,
        joint_names: Sequence[str] = DEFAULT_ALLEGRO_JOINT_NAMES,
        smoothing_alpha: float = 0.45,
        stale_timeout_s: float = 0.5,
    ):
        self.lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(16)
        self.upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(16)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 16 or len(set(self.joint_names)) != 16:
            raise ValueError("Allegro joint_names must contain 16 distinct names")
        if not np.all(np.isfinite(self.lower_limits)) or not np.all(np.isfinite(self.upper_limits)) or not np.all(self.upper_limits > self.lower_limits):
            raise ValueError("Allegro joint limits must be finite and ordered")
        if not 0.0 < float(smoothing_alpha) <= 1.0 or stale_timeout_s <= 0:
            raise ValueError("smoothing_alpha must be in (0, 1] and stale_timeout_s must be positive")
        self.smoothing_alpha = float(smoothing_alpha)
        self.stale_timeout_s = float(stale_timeout_s)
        self._targets = np.clip(np.zeros(16), self.lower_limits, self.upper_limits)
        self._curls = np.zeros(4, dtype=np.float64)
        self._timestamp = 0.0
        self._tracking_state = "waiting_for_hand"
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def calibration_state(self) -> str:
        # The legacy finger algorithm uses relative landmark geometry and has
        # no finger-specific calibration step; wrist calibration stays frozen.
        return "not_required_rule_based_fingers"

    def update_landmarks(self, landmarks: Sequence[Any], timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        try:
            curls = estimate_all_finger_curls(landmarks)
            raw_targets = curls_to_allegro_targets(curls, self.lower_limits, self.upper_limits)
        except (TypeError, ValueError, IndexError) as exc:
            return self.mark_lost(timestamp, f"invalid landmarks: {exc}")
        with self._lock:
            self._targets = self.smoothing_alpha * raw_targets + (1.0 - self.smoothing_alpha) * self._targets
            self._targets = np.clip(self._targets, self.lower_limits, self.upper_limits)
            self._curls = curls
            self._timestamp = timestamp
            self._tracking_state = "tracking"
            self._error = None
            return self._command_locked(valid=True)

    def mark_lost(self, timestamp_s: float | None = None, error: str | None = None) -> HandCommand:
        """Legacy safe behavior: retain last filtered fingers rather than open/jump."""
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            self._tracking_state = "lost_hold"
            self._error = error
            return self._command_locked(valid=False, timestamp=timestamp)

    def latest(self, timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            valid = self._tracking_state == "tracking" and timestamp - self._timestamp <= self.stale_timeout_s
            if not valid and self._tracking_state == "tracking":
                self._tracking_state = "stale_hold"
            return self._command_locked(valid=valid, timestamp=timestamp)

    def reset(self, timestamp_s: float | None = None) -> HandCommand:
        """Clear camera history while retaining the worker and calibration mode.

        The integrated ``R`` command must not reopen the camera or MediaPipe.
        It instead returns the hand source to the configured safe-open target
        until the next newly observed landmark frame arrives.
        """
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            self._targets = np.clip(np.zeros(16), self.lower_limits, self.upper_limits)
            self._curls = np.zeros(4, dtype=np.float64)
            self._timestamp = timestamp
            self._tracking_state = "waiting_for_hand"
            self._error = None
            return self._command_locked(valid=False, timestamp=timestamp)

    def _command_locked(self, valid: bool, timestamp: float | None = None) -> HandCommand:
        return HandCommand(
            joint_targets=self._targets.copy(), valid=valid,
            timestamp_s=self._timestamp if timestamp is None else timestamp,
            tracking_state=self._tracking_state, calibration_state=self.calibration_state,
            curls=self._curls.copy(), error=self._error,
            details={"algorithm": "legacy-curl"},
        )


class DexHandInputAdapter:
    """Latest-value camera source backed by the official dex vector solver.

    The adapter has the same narrow runtime contract as :class:`HandInputAdapter`:
    it produces a bounded 16-joint target, carries no wrist command, and becomes
    invalid after a finite timeout.  It deliberately does not use the legacy
    curl path if dex rejects a frame or is unavailable.
    """

    def __init__(
        self,
        retargeter: Any,
        *,
        lower_limits: Sequence[float] = DEFAULT_ALLEGRO_LOWER_LIMITS,
        upper_limits: Sequence[float] = DEFAULT_ALLEGRO_UPPER_LIMITS,
        joint_names: Sequence[str] = DEFAULT_ALLEGRO_JOINT_NAMES,
        stale_timeout_s: float = 0.5,
        expected_handedness: str = "Right",
        input_is_mirrored: bool = False,
        relative_motion: bool = True,
        relative_motion_gain: float = 1.0,
    ):
        self.retargeter = retargeter
        self.lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(16)
        self.upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(16)
        self.joint_names = tuple(joint_names)
        if (
            len(self.joint_names) != 16
            or len(set(self.joint_names)) != 16
            or not np.all(np.isfinite(self.lower_limits))
            or not np.all(np.isfinite(self.upper_limits))
            or not np.all(self.upper_limits > self.lower_limits)
            or stale_timeout_s <= 0
            or not np.isfinite(relative_motion_gain)
            or float(relative_motion_gain) <= 0.0
            or float(relative_motion_gain) > 3.0
        ):
            raise ValueError(
                "dex hand adapter requires finite 16-joint limits, a positive stale timeout, "
                "and relative_motion_gain in (0, 3]"
            )
        self.stale_timeout_s = float(stale_timeout_s)
        self.expected_handedness = str(expected_handedness)
        self.input_is_mirrored = bool(input_is_mirrored)
        self.relative_motion = bool(relative_motion)
        self.relative_motion_gain = float(relative_motion_gain)
        self._targets = np.clip(np.zeros(16), self.lower_limits, self.upper_limits)
        self._safe_open_targets = self._targets.copy()
        self._raw_targets: np.ndarray | None = None
        self._relative_reference_targets: np.ndarray | None = None
        self._timestamp = 0.0
        self._tracking_state = "waiting_for_hand"
        self._error: str | None = None
        self._details: dict[str, Any] = {
            "algorithm": "dex-retargeting",
            "expected_handedness": self.expected_handedness,
            "input_is_mirrored": self.input_is_mirrored,
        }
        self._lock = threading.Lock()

    @property
    def smoothing_alpha(self) -> None:
        # The official config's low-pass alpha is the sole dex smoothing layer.
        return None

    @property
    def calibration_state(self) -> str:
        if not self.relative_motion:
            return "absolute official dex vector target"
        if self._relative_reference_targets is None:
            return "waiting to capture camera-relative dex neutral"
        return "camera-relative dex neutral captured"

    def recenter_relative_motion(self) -> bool:
        """Make the newest valid official dex target the camera-relative neutral.

        This does not alter the official solver, MuJoCo state, or joint limits.
        It simply maps later camera-pose changes around Allegro's safe-open
        target, so an operator can work from a comfortable open hand.
        """

        if not self.relative_motion:
            return False
        with self._lock:
            if self._raw_targets is None:
                return False
            self._relative_reference_targets = self._raw_targets.copy()
            self._targets = self._safe_open_targets.copy()
            self._details = {
                **self._details,
                "relative_motion": True,
                "relative_motion_gain": self.relative_motion_gain,
                "relative_reference_event": "recentered",
            }
            return True

    def update_world_landmarks(
        self,
        landmarks: Sequence[Any],
        handedness: str,
        timestamp_s: float | None = None,
    ) -> HandCommand:
        """Run one valid MediaPipe world frame through the named dex mapping."""

        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        try:
            from .dex_retargeting import mediapipe_world_to_mano

            mano = mediapipe_world_to_mano(
                landmarks,
                handedness=handedness,
                expected_handedness=self.expected_handedness,
                input_is_mirrored=self.input_is_mirrored,
            )
            result = self.retargeter.retarget(mano)
            targets = np.asarray(result.joint_targets_rad, dtype=np.float64)
            if targets.shape != (16,) or not np.all(np.isfinite(targets)):
                raise ValueError("dex returned an invalid 16-joint target")
            targets = np.clip(targets, self.lower_limits, self.upper_limits)
        except Exception as exc:
            return self.mark_lost(timestamp, f"dex frame rejected: {type(exc).__name__}: {exc}")
        with self._lock:
            raw_targets = targets.copy()
            relative_reference_event: str | None = None
            if self.relative_motion:
                if self._relative_reference_targets is None:
                    self._relative_reference_targets = raw_targets.copy()
                    relative_reference_event = "captured_first_valid_frame"
                targets = self._safe_open_targets + self.relative_motion_gain * (
                    raw_targets - self._relative_reference_targets
                )
                targets = np.clip(targets, self.lower_limits, self.upper_limits)
            self._raw_targets = raw_targets
            self._targets = targets
            self._timestamp = timestamp
            self._tracking_state = "tracking"
            self._error = None
            self._details = {
                "algorithm": "dex-retargeting",
                "expected_handedness": self.expected_handedness,
                "observed_handedness": str(handedness),
                "input_is_mirrored": self.input_is_mirrored,
                "mano_landmarks_m": mano,
                "human_indices": np.asarray(result.human_indices, dtype=np.intp),
                "ref_value_m": np.asarray(result.ref_value, dtype=np.float64),
                "raw_dex_qpos": np.asarray(result.raw_qpos, dtype=np.float64),
                "retargeting_joint_names": list(result.retargeting_joint_names),
                "mapped_allegro_targets_rad": raw_targets,
                "clipped_allegro_targets_rad": targets.copy(),
                "relative_motion": self.relative_motion,
                "relative_motion_gain": self.relative_motion_gain,
                "relative_reference_event": relative_reference_event,
            }
            return self._command_locked(valid=True)

    def update_landmarks(self, _landmarks: Sequence[Any], timestamp_s: float | None = None) -> HandCommand:
        """Reject normalized image points; dex requires 3-D world landmarks."""

        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        return self.mark_lost(timestamp, "dex requires MediaPipe world landmarks, not normalized image landmarks")

    def mark_lost(self, timestamp_s: float | None = None, error: str | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            self._tracking_state = "lost_hold"
            self._error = error
            age = max(0.0, timestamp - self._timestamp)
            # A transient missed detector frame should not instantly snap the
            # hand open. Hold only the most recent valid target for the same
            # bounded timeout used by ``latest``; after that the gesture
            # layer receives ``camera_valid=False`` and moves to safe-open.
            valid = age <= self.stale_timeout_s and self._timestamp > 0.0
            self._details = {
                **self._details,
                "rejected_frame_reason": error,
                "loss_policy": "hold_last_valid_then_safe_open",
                "last_valid_age_s": age,
            }
            return self._command_locked(valid=valid, timestamp=timestamp)

    def latest(self, timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            age = max(0.0, timestamp - self._timestamp)
            valid = self._timestamp > 0.0 and age <= self.stale_timeout_s
            if not valid:
                self._tracking_state = "stale_safe_open"
                self._details = {
                    **self._details,
                    "rejected_frame_reason": "dex camera frame stale",
                    "loss_policy": "hold_last_valid_then_safe_open",
                    "last_valid_age_s": age,
                }
            return self._command_locked(valid=valid, timestamp=timestamp)

    def reset(self, timestamp_s: float | None = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        try:
            self.retargeter.reset()
        except Exception as exc:
            raise RuntimeError(f"dex history reset failed: {type(exc).__name__}: {exc}") from exc
        with self._lock:
            self._targets = np.clip(np.zeros(16), self.lower_limits, self.upper_limits)
            self._raw_targets = None
            self._relative_reference_targets = None
            self._timestamp = timestamp
            self._tracking_state = "waiting_for_hand"
            self._error = None
            self._details = {
                "algorithm": "dex-retargeting",
                "expected_handedness": self.expected_handedness,
                "input_is_mirrored": self.input_is_mirrored,
                "relative_motion": self.relative_motion,
                "relative_motion_gain": self.relative_motion_gain,
                "history_reset": True,
            }
            return self._command_locked(valid=False, timestamp=timestamp)

    def _command_locked(self, valid: bool, timestamp: float | None = None) -> HandCommand:
        return HandCommand(
            joint_targets=self._targets.copy(),
            valid=valid,
            timestamp_s=self._timestamp if timestamp is None else timestamp,
            tracking_state=self._tracking_state,
            calibration_state=self.calibration_state,
            curls=np.zeros(4, dtype=np.float64),
            error=self._error,
            details=dict(self._details),
        )


class CameraHandWorker:
    """One asynchronous MediaPipe producer and its optional non-blocking UI.

    The worker owns exactly one ``VideoCapture`` and one MediaPipe Hands
    instance.  Its OpenCV window is deliberately a view of that same stream;
    it never opens a second capture or performs a second detector pass merely
    to draw diagnostics.  A caller-provided status snapshot lets the UI show
    the main loop's active gesture source without giving this thread write
    access to robot actions.
    """

    def __init__(
        self,
        adapter: Any,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        display: bool = True,
        mirror_input: bool = True,
        keypress_callback: Callable[[int], None] | None = None,
        status_provider: Callable[[], dict[str, Any]] | None = None,
        window_name: str = "Panda + Allegro camera hand input",
        display_text: bool = True,
    ):
        self.adapter = adapter
        self.camera_index, self.width, self.height, self.display = camera_index, width, height, display
        # False keeps only the drawn landmarks, for callers that embed the
        # frame in their own UI (``latest_frame``) instead of this window.
        self.display_text = bool(display_text)
        self.mirror_input = bool(mirror_input)
        self.keypress_callback = keypress_callback
        self.status_provider = status_provider
        self.window_name = str(window_name)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._window_close_notified = False
        self._last_display_timestamp: float | None = None
        self._display_fps = 0.0
        self._last_dex_update_timestamp: float | None = None
        self._status_lock = threading.Lock()
        # MediaPipe acquisition stays on its worker, but HighGUI is serviced
        # by the simulation thread. Windows HighGUI has process-global state;
        # calling imshow / waitKey from both this worker and the tri-view loop
        # causes intermittent black or corrupt simulator frames.
        self._display_lock = threading.Lock()
        self._latest_display_frame: np.ndarray | None = None
        self._latest_display_timestamp: float | None = None
        self._capture_lock = threading.Lock()
        self._capture: Any | None = None
        self._capture_backend = "not_started"
        self._frames_read = 0
        self._frames_processed = 0
        self._read_failures = 0
        self._last_frame_timestamp: float | None = None
        self._detected_handedness: tuple[str, ...] = ()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("CameraHandWorker is already running")
        self._thread = threading.Thread(target=self._run, name="camera-hand-input", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        # ``VideoCapture.read`` can block indefinitely on a disconnected or
        # privacy-locked Windows camera.  Releasing the same, single capture
        # from the shutdown path gives that read a chance to return before we
        # join the worker; it never opens a replacement capture.
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Do not silently permit a second worker to compete for the
                # camera if a backend has blocked inside ``read``.
                self.error = self.error or "camera worker did not stop within 2 seconds"
                self.adapter.mark_lost(error=self.error)
            else:
                self._thread = None
        if self.display:
            try:
                import cv2
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        """Small read-only telemetry for the simulator log and camera UI audit."""

        with self._capture_lock:
            capture_open = self._capture is not None
        with self._status_lock:
            return {
                "running": self.running,
                "frames_read": self._frames_read,
                "frames_processed": self._frames_processed,
                "read_failures": self._read_failures,
                "last_frame_timestamp_s": self._last_frame_timestamp,
                "detected_handedness": list(self._detected_handedness),
                "display_fps": self._display_fps,
                "display_frame_available": self._latest_display_frame is not None,
                "display_frame_timestamp_s": self._latest_display_timestamp,
                "window_close_requested": self._window_close_notified,
                "capture_open": capture_open,
                "capture_backend": self._capture_backend,
                "error": self.error,
            }

    @staticmethod
    def _open_camera_capture(cv2: Any, camera_index: int) -> tuple[Any, str]:
        """Open one Windows camera with a bounded, reliable backend choice.

        OpenCV's automatic backend can block inside Media Foundation before it
        publishes a capture object.  DirectShow opens the same UVC devices
        promptly on this Windows workstation.  Only if DirectShow cannot open
        the selected index do we release it and try OpenCV's normal backend;
        at no time are two captures active.
        """

        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture, "CAP_DSHOW"
            capture.release()
        return cv2.VideoCapture(camera_index), "CAP_ANY"

    def _runtime_status(self) -> dict[str, Any]:
        """Return a detached best-effort status snapshot for overlay drawing."""

        if self.status_provider is None:
            return {}
        try:
            provided = self.status_provider()
        except Exception as exc:
            return {"status_error": f"status provider failed: {type(exc).__name__}: {exc}"}
        return dict(provided) if isinstance(provided, dict) else {}

    @staticmethod
    def _put_lines(cv2: Any, frame: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
        """Draw compact, readable diagnostic text without changing frame size."""

        y = 26
        for text, color in lines:
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y += 23

    def _draw_display(
        self,
        cv2: Any,
        frame: np.ndarray,
        command: HandCommand,
        *,
        handedness: str | None,
        timestamp: float,
    ) -> None:
        """Render camera, detector, dex, and gesture status for one real frame."""

        if self._last_display_timestamp is not None:
            elapsed = timestamp - self._last_display_timestamp
            if elapsed > 1.0e-6:
                instantaneous = 1.0 / elapsed
                with self._status_lock:
                    self._display_fps = (
                        instantaneous if self._display_fps <= 0.0
                        else 0.25 * instantaneous + 0.75 * self._display_fps
                    )
        self._last_display_timestamp = timestamp

        details = command.details if isinstance(command.details, dict) else {}
        # ``mark_lost`` deliberately retains the most recent target for a
        # bounded grace period.  It must be displayed as a hold, not as a
        # new retarget() result from the rejected camera frame.
        is_dex_frame = (
            details.get("algorithm") == "dex-retargeting"
            and command.valid
            and not details.get("rejected_frame_reason")
        )
        if is_dex_frame:
            self._last_dex_update_timestamp = timestamp
        runtime = self._runtime_status()
        hand_mode = str(runtime.get("hand_mode", "dex" if details.get("algorithm") == "dex-retargeting" else "legacy-curl"))
        hand_source = str(runtime.get("hand_source", "camera" if command.valid else "safe-open/hold"))
        active_override = runtime.get("active_override")
        mode_label = hand_source if active_override else hand_mode
        detection = "VALID" if command.valid else "LOST / HOLD"
        detection_color = (40, 220, 40) if command.valid else (30, 170, 255)
        hand_label = handedness if handedness else "none"

        dex_line = None
        if hand_mode == "dex":
            if is_dex_frame:
                dex_line = ("DEX ACTIVE: retarget() updated", (40, 220, 40))
            elif "expected right hand" in str(details.get("rejected_frame_reason", "")).lower():
                dex_line = ("DEX WAITING: show your physical RIGHT hand", (30, 170, 255))
            elif self._last_dex_update_timestamp is not None and timestamp - self._last_dex_update_timestamp <= 1.0:
                dex_line = ("DEX HOLD: waiting for next valid frame", (30, 170, 255))
            else:
                dex_line = ("DEX WAITING: no valid output", (30, 170, 255))

        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"Camera {self._display_fps:5.1f} FPS | detected hand: {hand_label}", (230, 230, 230)),
            (f"tracking: {detection} ({command.tracking_state})", detection_color),
            (f"hand mode: {hand_mode} | active: {mode_label}", (255, 230, 80) if active_override else (220, 220, 220)),
        ]
        if dex_line is not None:
            lines.append(dex_line)
        if hand_mode == "dex" and bool(details.get("relative_motion", False)):
            calibration_event = details.get("relative_reference_event")
            calibration_text = "DEX RELATIVE: C re-centres your current hand"
            if calibration_event == "captured_first_valid_frame":
                calibration_text = "DEX RELATIVE: neutral captured; move your fingers"
            lines.append((calibration_text, (180, 235, 180)))
        if command.error:
            lines.append((f"input: {command.error[:88]}", (40, 120, 255)))
        lines.append(("C re-centre dex | R reset | Q quit | close window = safe quit", (210, 210, 210)))
        self._put_lines(cv2, frame, lines)

    def _show_frame(self, cv2: Any, frame: np.ndarray) -> None:
        """Display once, route R/Q, and turn a user window close into Q safely."""

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1)
        if key >= 0 and self.keypress_callback is not None:
            self.keypress_callback(int(key) & 0xFF)
        try:
            visible = float(cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE))
        except Exception:
            visible = 1.0
        if visible < 1.0 and not self._window_close_notified:
            self._window_close_notified = True
            if self.keypress_callback is not None:
                self.keypress_callback(ord("q"))

    def latest_frame(self) -> tuple[np.ndarray | None, float | None]:
        """Detached copy of the latest annotated BGR frame, for an embedded preview."""

        with self._display_lock:
            frame = None if self._latest_display_frame is None else self._latest_display_frame.copy()
            return frame, self._latest_display_timestamp

    def show_latest(self) -> dict[str, Any]:
        """Show the already-annotated camera frame from the main UI thread.

        The worker owns one capture and one MediaPipe detector, then publishes
        a detached display frame.  Calling this from the same thread as the
        tri-view compositor guarantees that Windows HighGUI never has two
        competing event loops.  It never reads the camera or invokes the hand
        adapter, so it cannot affect robot actions.
        """

        if not self.display:
            return {"enabled": False, "reason": "display_disabled"}
        with self._display_lock:
            frame = None if self._latest_display_frame is None else self._latest_display_frame.copy()
            timestamp = self._latest_display_timestamp
        if frame is None:
            return {"enabled": True, "rendered": False, "reason": "no_camera_frame"}
        try:
            import cv2
            self._show_frame(cv2, frame)
            return {
                "enabled": True,
                "rendered": True,
                "frame_timestamp_s": timestamp,
                "closed": bool(self._window_close_notified),
            }
        except Exception as exc:
            # Camera visualization is optional. Retaining acquisition and
            # hand safety is preferable to stopping the robot on a GUI fault.
            self.error = self.error or f"camera display failed: {type(exc).__name__}: {exc}"
            return {"enabled": True, "rendered": False, "error": self.error}

    @staticmethod
    def _select_world_hand_index(
        labels: Sequence[str | None], *, expected_handedness: str, input_is_mirrored: bool,
    ) -> int:
        """Choose the physical dex hand when MediaPipe sees both hands.

        The selection is made before invoking the solver.  If the configured
        physical hand is absent, index zero is returned so the existing
        handedness gate emits its precise rejected-frame diagnostic rather
        than silently retargeting the wrong hand.
        """

        expected = str(expected_handedness).strip().lower()
        for index, raw_label in enumerate(labels):
            observed = str(raw_label or "").strip().lower()
            if input_is_mirrored and observed in {"left", "right"}:
                observed = "left" if observed == "right" else "right"
            if observed == expected:
                return index
        return 0

    @staticmethod
    def _physical_handedness(label: str | None, *, detector_input_is_mirrored: bool) -> str | None:
        """Turn MediaPipe's selfie-image handedness into a physical label."""

        if label is None:
            return None
        value = str(label).strip()
        if value.lower() not in {"left", "right"}:
            return value
        # MediaPipe documents its label as valid for mirrored / selfie input.
        # A raw camera frame therefore needs a left-right swap before the dex
        # right-hand gate is evaluated.
        if detector_input_is_mirrored:
            return value.title()
        return "Right" if value.lower() == "left" else "Left"

    @staticmethod
    def _world_landmarks_at(world_sets: Sequence[Any], selected_index: int) -> Sequence[Any] | None:
        """Return the 3-D landmark set paired with the selected 2-D hand.

        MediaPipe reports image landmarks, world landmarks, and handedness in
        matching list order.  Keeping this lookup explicit prevents a
        right-hand dex selection from accidentally passing index zero's world
        landmarks when two hands are visible.
        """

        if selected_index < 0 or selected_index >= len(world_sets):
            return None
        selected = world_sets[selected_index]
        landmarks = getattr(selected, "landmark", None)
        return landmarks if landmarks is not None else None

    def _run(self) -> None:  # pragma: no cover - requires a physical camera / GUI
        capture = None
        hands = None
        try:
            import cv2
            import mediapipe as mp
            capture, backend = self._open_camera_capture(cv2, self.camera_index)
            self._capture_backend = backend
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
            with self._capture_lock:
                self._capture = capture
            hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                              min_detection_confidence=0.60, min_tracking_confidence=0.60)
            while not self._stop.is_set():
                ok, frame = capture.read()
                timestamp = time.monotonic()
                if not ok:
                    with self._status_lock:
                        self._read_failures += 1
                    self.adapter.mark_lost(timestamp, "camera read failed")
                    time.sleep(0.02)
                    continue
                with self._status_lock:
                    self._frames_read += 1
                    self._last_frame_timestamp = timestamp
                # MediaPipe's handedness classifier assumes a selfie-style
                # mirrored image. Canonicalise raw Windows webcam frames by
                # default. ``mirror_input=False`` is reserved for a driver
                # that already supplies that same mirrored presentation.
                if self.mirror_input:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                rgb.flags.writeable = True
                landmark_sets = list(getattr(result, "multi_hand_landmarks", None) or ())
                handedness_sets = list(getattr(result, "multi_handedness", None) or ())
                labels: list[str | None] = []
                for handedness_set in handedness_sets:
                    classifications = getattr(handedness_set, "classification", ())
                    label = getattr(classifications[0], "label", None) if classifications else None
                    labels.append(
                        self._physical_handedness(
                            label,
                            detector_input_is_mirrored=self.mirror_input,
                        )
                    )
                with self._status_lock:
                    self._frames_processed += 1
                    self._detected_handedness = tuple(label for label in labels if label)
                frame_updater = getattr(self.adapter, "update_hand_frame", None)
                if landmark_sets and callable(frame_updater):
                    # Adapters that need both landmark sets of one detection
                    # (camera-driven arm + fingers) receive them together.
                    selected_index = self._select_world_hand_index(
                        labels,
                        expected_handedness=getattr(self.adapter, "expected_handedness", "Right"),
                        input_is_mirrored=False,
                    )
                    landmarks = landmark_sets[selected_index]
                    label = labels[selected_index] if selected_index < len(labels) else None
                    world_landmarks = self._world_landmarks_at(
                        getattr(result, "multi_hand_world_landmarks", None) or (), selected_index
                    )
                    if not label:
                        command = self.adapter.mark_lost(timestamp, "MediaPipe handedness label unavailable")
                    else:
                        command = frame_updater(
                            landmarks.landmark, world_landmarks, str(label), timestamp, frame.shape[:2],
                            detected_hands=tuple(labels),
                        )
                elif landmark_sets:
                    world_updater = getattr(self.adapter, "update_world_landmarks", None)
                    selected_index = 0
                    if callable(world_updater):
                        selected_index = self._select_world_hand_index(
                            labels,
                            expected_handedness=getattr(self.adapter, "expected_handedness", "Right"),
                            # ``labels`` were normalized to physical hands
                            # above, so selection never depends on whether a
                            # particular camera driver mirrored its output.
                            input_is_mirrored=False,
                        )
                    landmarks = landmark_sets[selected_index]
                    label = labels[selected_index] if selected_index < len(labels) else None
                    if callable(world_updater):
                        world_sets = getattr(result, "multi_hand_world_landmarks", None)
                        world_landmarks = self._world_landmarks_at(world_sets or (), selected_index)
                        if not handedness_sets or world_landmarks is None:
                            command = self.adapter.mark_lost(
                                timestamp, "dex requires MediaPipe world landmarks and handedness"
                            )
                        else:
                            if not label:
                                command = self.adapter.mark_lost(timestamp, "MediaPipe handedness label unavailable")
                            else:
                                command = world_updater(world_landmarks, str(label), timestamp)
                    else:
                        command = self.adapter.update_landmarks(landmarks.landmark, timestamp)
                else:
                    landmarks = None
                    command = self.adapter.mark_lost(timestamp)
                if self.display:
                    for landmarks_to_draw in landmark_sets:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, landmarks_to_draw, mp.solutions.hands.HAND_CONNECTIONS
                        )
                    hand_label = ", ".join(label for label in labels if label) or None
                    if self.display_text:
                        self._draw_display(cv2, frame, command, handedness=hand_label, timestamp=timestamp)
                    with self._display_lock:
                        self._latest_display_frame = np.ascontiguousarray(frame.copy())
                        self._latest_display_timestamp = timestamp
        except Exception as exc:
            if not self._stop.is_set():
                self.error = f"camera worker failed: {type(exc).__name__}: {exc}"
                self.adapter.mark_lost(error=self.error)
        finally:
            if hands is not None:
                hands.close()
            if capture is not None:
                capture.release()
            with self._capture_lock:
                if self._capture is capture:
                    self._capture = None
