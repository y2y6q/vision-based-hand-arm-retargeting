"""Official dex-retargeting DexPilot control for the 20-actuator Shadow Hand.

The solver runs in the same isolated sidecar used by Panda + Allegro, so this
module also works from the Python 3.8 LIBERO interpreter. DexPilot matches the
operator's thumb-to-finger vectors as well as wrist-to-fingertip vectors and
projects near-contact thumb pairs to contact, which is what makes pinch and
thumb opposition usable. The SpaceMouse owns the wrist pose, so the project
config keeps Shadow's WRJ2 / WRJ1 out of the optimization (held at zero).

The dex-urdf Shadow URDF and the MuJoCo Menagerie MJCF share joint names,
signs and limits (fingertip geometry agrees to 0.1 mm), so targets map by
name. The four distal tendon actuators take ``J2 + J1``.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

from .dex_retargeting import (
    OFFICIAL_DEX_VERSION,
    OFFICIAL_WHEEL,
    OFFICIAL_WHEEL_SHA256,
    DEX_SIDECAR_WORKER,
    DexFrameRejected,
    DexRetargetingUnavailable,
    _DexSidecar,
    _discover_sidecar_python,
    _sha256,
    mediapipe_world_to_mano,
)
from .hand_input import HandCommand
from .shadow_gripper import SHADOW_ACTUATOR_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHADOW_DEX_CONFIG = PROJECT_ROOT / "configs" / "dex_retargeting" / "shadow_hand_right_dexpilot.yml"
SHADOW_DEX_URDF_ROOT = PROJECT_ROOT / "third_party" / "dex_urdf"
SHADOW_DEX_JOINT_NAMES = frozenset((
    "WRJ2", "WRJ1",
    "FFJ4", "FFJ3", "FFJ2", "FFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1",
    "RFJ4", "RFJ3", "RFJ2", "RFJ1", "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
))
# Five fingertips -> 10 tip pairs + 5 wrist-to-tip vectors.
SHADOW_DEX_VECTOR_COUNT = 15


def _mainline_sidecar_python() -> Optional[Path]:
    """Use the mainline venv's base Conda env to find its ``dexretarget`` sibling.

    The LIBERO venv has a different base interpreter, so the Allegro rule
    (sibling of *this* interpreter's base) does not apply there.
    """
    config = PROJECT_ROOT / ".venv-robosuite" / "pyvenv.cfg"
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().lower() == "home" and value.strip():
            home = Path(value.strip())
            for base in (home, home.parent):
                candidate = base.parent / "dexretarget" / "python.exe"
                if candidate.is_file():
                    return candidate
    return None


def shadow_targets_from_dex(
    joint_names: Sequence[str],
    joint_values: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> np.ndarray:
    """Convert named dex Shadow joints into the 20 MuJoCo actuator targets."""

    values = dict(zip((str(name) for name in joint_names), np.asarray(joint_values, dtype=np.float64)))
    targets = np.zeros(len(SHADOW_ACTUATOR_NAMES), dtype=np.float64)
    for index, actuator in enumerate(SHADOW_ACTUATOR_NAMES):
        joint = actuator[len("rh_A_"):]
        if joint.startswith("WR"):
            targets[index] = 0.0  # wrist belongs to the SpaceMouse
        elif joint.endswith("J0"):
            targets[index] = values[joint[:2] + "J2"] + values[joint[:2] + "J1"]
        else:
            targets[index] = values[joint]
    if not np.all(np.isfinite(targets)):
        raise DexFrameRejected("dex Shadow mapping produced non-finite targets")
    return np.clip(targets, np.asarray(lower_limits, dtype=np.float64), np.asarray(upper_limits, dtype=np.float64))


class ShadowDexRetargeter:
    """Sidecar-backed official DexPilot solver for the right Shadow Hand."""

    def __init__(
        self,
        *,
        config_path: Path = SHADOW_DEX_CONFIG,
        urdf_root: Path = SHADOW_DEX_URDF_ROOT,
        sidecar_python: Optional[Path] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.urdf_root = Path(urdf_root)
        if not self.config_path.is_file():
            raise DexRetargetingUnavailable(f"Shadow dex config is missing: {self.config_path}")
        if not (self.urdf_root / "shadow_hand" / "shadow_hand_right.urdf").is_file():
            raise DexRetargetingUnavailable(f"Shadow dex URDF is missing under {self.urdf_root}")
        python_path = _discover_sidecar_python(sidecar_python) or _mainline_sidecar_python()
        if python_path is None:
            raise DexRetargetingUnavailable(
                "no dex sidecar Python found; set DEX_RETARGETING_PYTHON to the dexretarget interpreter"
            )
        self._lock = threading.RLock()
        self._sidecar: Optional[_DexSidecar] = _DexSidecar(python_path)
        try:
            metadata = self._sidecar.initialize(config_path=self.config_path, urdf_root=self.urdf_root)
            if metadata.get("package") != "dex-retargeting" or metadata.get("version") != OFFICIAL_DEX_VERSION:
                raise DexRetargetingUnavailable("dex sidecar did not report official dex-retargeting 0.4.6")
            if metadata.get("wheel_sha256") != OFFICIAL_WHEEL_SHA256:
                raise DexRetargetingUnavailable("dex sidecar wheel SHA-256 does not match the checked-in official wheel")
            if metadata.get("config_sha256") != _sha256(self.config_path):
                raise DexRetargetingUnavailable("dex sidecar config SHA-256 does not match the Shadow config")
            indices = np.asarray(metadata.get("human_indices"), dtype=np.intp)
            names = tuple(str(name) for name in metadata.get("joint_names", ()))
            if indices.shape != (2, SHADOW_DEX_VECTOR_COUNT):
                raise DexRetargetingUnavailable(f"Shadow DexPilot must expose 2x15 human indices, got {indices.shape}")
            if len(names) != len(SHADOW_DEX_JOINT_NAMES) or set(names) != SHADOW_DEX_JOINT_NAMES:
                raise DexRetargetingUnavailable(f"Shadow dex joint names do not match the Shadow URDF: {names}")
        except Exception as exc:
            self.close()
            if isinstance(exc, DexRetargetingUnavailable):
                raise
            raise DexRetargetingUnavailable(
                f"Shadow dex sidecar failed to initialize: {type(exc).__name__}: {exc}"
            ) from exc
        self.human_indices = indices
        self.joint_names = names
        self.python_path = python_path

    def retarget(self, mano_landmarks_m: Sequence[Sequence[float]]) -> np.ndarray:
        joints = np.asarray(mano_landmarks_m, dtype=np.float64)
        if joints.shape != (21, 3) or not np.all(np.isfinite(joints)):
            raise DexFrameRejected("dex retargeting requires a finite 21x3 MANO landmark array")
        with self._lock:
            if self._sidecar is None:
                raise DexFrameRejected("Shadow dex sidecar is closed")
            try:
                result = self._sidecar.request("retarget", mano_landmarks=joints.tolist())
            except Exception as exc:
                raise DexFrameRejected(f"official dex retarget call failed: {type(exc).__name__}: {exc}") from exc
        if tuple(str(name) for name in result.get("joint_names", ())) != self.joint_names:
            raise DexFrameRejected("dex sidecar changed output joint order after initialization")
        raw = np.asarray(result.get("raw_qpos"), dtype=np.float64).reshape(-1)
        if raw.shape != (len(self.joint_names),) or not np.all(np.isfinite(raw)):
            raise DexFrameRejected("official dex returned non-finite or incorrectly sized Shadow output")
        return raw

    def reset(self) -> None:
        with self._lock:
            if self._sidecar is not None:
                self._sidecar.request("reset")

    def close(self) -> None:
        with self._lock:
            if self._sidecar is not None:
                try:
                    self._sidecar.close()
                finally:
                    self._sidecar = None

    def metadata(self) -> dict[str, Any]:
        return {
            "package": "dex-retargeting",
            "version": OFFICIAL_DEX_VERSION,
            "runtime": "sidecar",
            "algorithm": "official DexPilot (thumb-finger pair + wrist-tip vectors)",
            "config_path": str(self.config_path),
            "config_sha256": _sha256(self.config_path),
            "urdf_root": str(self.urdf_root),
            "wheel_path": str(OFFICIAL_WHEEL),
            "worker_path": str(DEX_SIDECAR_WORKER),
            "runtime_python": str(self.python_path),
            "human_indices": self.human_indices.tolist(),
            "retargeting_joint_names": list(self.joint_names),
            "fixed_joints": ["WRJ2", "WRJ1"],
        }


class ShadowDexHandInput:
    """Camera-worker adapter: MediaPipe world landmarks -> 20 Shadow targets.

    Targets are absolute, not relative to a captured neutral: DexPilot's
    pinch projection depends on real thumb-to-finger distances. A rejected
    frame holds the last valid target for ``stale_timeout_s``; after that the
    gesture layer receives ``valid=False`` and returns to safe-open.

    ``unmirror_world_x`` undoes the horizontal flip the camera worker applies
    before MediaPipe. A flipped image yields mirror-image 3-D landmarks, which
    would hand a left-hand geometry to the right-hand solver.
    """

    def __init__(
        self,
        retargeter: Any,
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        *,
        stale_timeout_s: float = 0.5,
        expected_handedness: str = "Right",
        unmirror_world_x: bool = True,
    ) -> None:
        self.retargeter = retargeter
        self.lower_limits = np.asarray(lower_limits, dtype=np.float64).reshape(20)
        self.upper_limits = np.asarray(upper_limits, dtype=np.float64).reshape(20)
        if (not np.all(np.isfinite(self.lower_limits)) or not np.all(np.isfinite(self.upper_limits))
                or not np.all(self.upper_limits > self.lower_limits) or stale_timeout_s <= 0):
            raise ValueError("Shadow dex input needs finite ordered limits and a positive stale timeout")
        self.joint_names = SHADOW_ACTUATOR_NAMES
        self.stale_timeout_s = float(stale_timeout_s)
        self.expected_handedness = str(expected_handedness)
        self.unmirror_world_x = bool(unmirror_world_x)
        self._safe_open = np.clip(np.zeros(20), self.lower_limits, self.upper_limits)
        self._targets = self._safe_open.copy()
        self._timestamp = 0.0
        self._tracking_state = "waiting_for_hand"
        self._error: Optional[str] = None
        self._details: dict[str, Any] = {"algorithm": "dex-retargeting DexPilot"}
        self._lock = threading.Lock()

    @property
    def smoothing_alpha(self) -> None:
        # The official config's low_pass_alpha is the only smoothing layer.
        return None

    @property
    def calibration_state(self) -> str:
        return "absolute official DexPilot target"

    def _command_locked(self, valid: bool, timestamp: Optional[float] = None) -> HandCommand:
        return HandCommand(
            joint_targets=self._targets.copy(), valid=valid,
            timestamp_s=self._timestamp if timestamp is None else timestamp,
            tracking_state=self._tracking_state, calibration_state=self.calibration_state,
            curls=np.zeros(4, dtype=np.float64), error=self._error, details=dict(self._details),
        )

    def update_world_landmarks(
        self, landmarks: Sequence[Any], handedness: str, timestamp_s: Optional[float] = None,
    ) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        try:
            points = np.asarray(
                [(point.x, point.y, point.z) if hasattr(point, "x") else point for point in landmarks],
                dtype=np.float64,
            )
            if points.shape != (21, 3):
                raise DexFrameRejected("dex requires 21 world landmarks")
            if self.unmirror_world_x:
                points = points * np.array((-1.0, 1.0, 1.0))
            mano = mediapipe_world_to_mano(
                points, handedness=handedness, expected_handedness=self.expected_handedness,
            )
            raw = self.retargeter.retarget(mano)
            targets = shadow_targets_from_dex(
                self.retargeter.joint_names, raw, self.lower_limits, self.upper_limits
            )
        except Exception as exc:
            return self.mark_lost(timestamp, f"dex frame rejected: {type(exc).__name__}: {exc}")
        with self._lock:
            self._targets = targets
            self._timestamp = timestamp
            self._tracking_state = "tracking"
            self._error = None
            self._details = {
                "algorithm": "dex-retargeting DexPilot",
                "observed_handedness": str(handedness),
                "unmirror_world_x": self.unmirror_world_x,
                "raw_dex_qpos": raw,
                "retargeting_joint_names": list(self.retargeter.joint_names),
            }
            return self._command_locked(valid=True)

    def update_landmarks(self, _landmarks: Sequence[Any], timestamp_s: Optional[float] = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        return self.mark_lost(timestamp, "dex requires MediaPipe world landmarks, not normalized image landmarks")

    def mark_lost(self, timestamp_s: Optional[float] = None, error: Optional[str] = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            self._tracking_state = "lost_hold"
            self._error = error
            valid = self._timestamp > 0.0 and timestamp - self._timestamp <= self.stale_timeout_s
            return self._command_locked(valid=valid, timestamp=timestamp)

    def latest(self, timestamp_s: Optional[float] = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        with self._lock:
            valid = self._timestamp > 0.0 and timestamp - self._timestamp <= self.stale_timeout_s
            if not valid and self._tracking_state == "tracking":
                self._tracking_state = "stale_safe_open"
            return self._command_locked(valid=valid, timestamp=timestamp)

    def reset(self, timestamp_s: Optional[float] = None) -> HandCommand:
        timestamp = time.monotonic() if timestamp_s is None else float(timestamp_s)
        self.retargeter.reset()
        with self._lock:
            self._targets = self._safe_open.copy()
            self._timestamp = 0.0
            self._tracking_state = "waiting_for_hand"
            self._error = None
            return self._command_locked(valid=False, timestamp=timestamp)
