"""Convert six raw motion axes into bounded OSC world-frame wrist deltas."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config import WristConfig


_WORLD_UP = np.array((0.0, 0.0, 1.0), dtype=np.float64)
_EPSILON = 1.0e-9


def _proper_rotation(value: Any, *, name: str) -> np.ndarray:
    """Return a finite proper 3x3 rotation or raise a clear configuration error."""
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite proper 3x3 rotation") from exc
    if (
        not np.all(np.isfinite(matrix))
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError(f"{name} must be a finite proper 3x3 rotation")
    return matrix


def _rotation_from_wxyz(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a validated MuJoCo ``wxyz`` camera quaternion to world axes."""
    try:
        w, x, y, z = (float(value) for value in quaternion_wxyz)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera quaternion must contain four finite values") from exc
    quaternion = np.array((w, x, y, z), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(quaternion)) or norm <= _EPSILON:
        raise ValueError("camera quaternion must contain four finite values")
    w, x, y, z = quaternion / norm
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def camera_frame_to_world_rotation(world_from_camera: Any) -> np.ndarray:
    """Build the project control basis from a camera's real world extrinsics.

    MuJoCo camera axes are ``(+X right, +Y up, -Z looking forward)``. For
    wrist teleoperation we retain screen-right as the first axis, make the
    second axis point through the primary view's horizontal depth, and force
    the third axis to world vertical. This produces an orthonormal proper
    rotation whose columns mean ``[screen right, view depth, world up]``.
    It works for Panda and Jaco because it uses only the selected camera's
    current extrinsics, never a robot-specific world-axis permutation.
    """
    matrix = _proper_rotation(world_from_camera, name="world_from_camera")
    screen_right = matrix[:, 0]
    horizontal_right = screen_right - _WORLD_UP * float(np.dot(screen_right, _WORLD_UP))
    right_norm = float(np.linalg.norm(horizontal_right))
    if right_norm <= _EPSILON:
        raise ValueError("camera right axis is parallel to world up")
    right = horizontal_right / right_norm
    depth = np.cross(_WORLD_UP, right)
    depth_norm = float(np.linalg.norm(depth))
    if depth_norm <= _EPSILON:  # defensive: implied by a valid ``right`` above
        raise ValueError("camera depth axis cannot be constructed")
    depth /= depth_norm
    camera_forward = -matrix[:, 2]
    horizontal_forward = camera_forward - _WORLD_UP * float(np.dot(camera_forward, _WORLD_UP))
    if float(np.linalg.norm(horizontal_forward)) <= _EPSILON:
        raise ValueError("camera look direction is parallel to world up")
    # An upright MuJoCo camera has this sign already. Reject a flipped or
    # rolled-over camera instead of silently inverting screen-right control.
    if float(np.dot(depth, horizontal_forward)) <= 0.0:
        raise ValueError("camera right/depth axes are inconsistent with its look direction")
    return _proper_rotation(np.column_stack((right, depth, _WORLD_UP)), name="camera control basis")


class SixDofMapper:
    """Stateful filtering with optional camera-relative world mapping.

    The mapper never integrates an absolute pose. It accepts only six motion
    axes and therefore cannot let SpaceMouse buttons alter hand / gripper
    actions. Invalid samples clear filter state and return exact zero.

    In ``control_frame='camera'`` mode, :meth:`set_camera_from_sim` should be
    called after the primary camera has been configured. A validated
    configuration quaternion provides a safe deterministic fallback before
    the first viewer render; the live camera matrix supersedes it immediately
    once bound.
    """

    def __init__(self, config: WristConfig):
        if not isinstance(config, WristConfig):
            raise TypeError("config must be a validated WristConfig")
        self.config = config
        self._raw_ranges = np.asarray(config.raw_ranges, dtype=np.float64)
        self._axis_order = np.asarray(config.axis_order, dtype=np.intp)
        self._axis_inversion = np.asarray(config.axis_inversion, dtype=np.float64)
        self._device_to_world = np.asarray(config.device_to_world, dtype=np.float64)
        self._deadzones = np.array([config.translation_deadzone] * 3 + [config.rotation_deadzone] * 3)
        self._release_deadzones = np.array(
            [config.translation_release_deadzone] * 3 + [config.rotation_release_deadzone] * 3
        )
        self._filtered = np.zeros(6, dtype=np.float64)
        self._axis_active = np.zeros(6, dtype=bool)
        self._neutral_frames = 0
        self._last_diagnostics: dict[str, Any] = {
            "valid": False,
            "calibrated_axes": [0.0] * 6,
            "deadzone_axes": [0.0] * 6,
            "ema_axes": [0.0] * 6,
            "final_osc_delta": [0.0] * 6,
            "axis_active": [False] * 6,
            "neutral_consecutive_frames": 0,
            "neutral_locked": True,
        }
        self._control_rotation = self._device_to_world.copy()
        self._control_camera_name: str | None = None
        self._control_rotation_source = "configured_device_to_world"
        if config.control_frame == "camera":
            # The tri-view's fixed front quaternion is the exact intended
            # camera pose. A later live binding reads data.cam_xmat instead.
            self.set_camera_extrinsics(
                _rotation_from_wxyz(config.tri_view.front_quat_wxyz),
                camera_name=config.tri_view.front_camera_name,
                source="configured_tri_view_quaternion",
            )

    @property
    def control_rotation(self) -> np.ndarray:
        """Detached world-from-control basis used by translation and rotation."""
        return self._control_rotation.copy()

    def describe_control_frame(self) -> dict[str, Any]:
        """JSON-ready evidence of the current world mapping source."""
        return {
            "control_frame": self.config.control_frame,
            "coordinate_frame": self.config.coordinate_frame,
            "primary_camera_name": self._control_camera_name,
            "world_from_control_rotation": self._control_rotation.tolist(),
            "source": self._control_rotation_source,
            "basis_axes": ["screen_right", "view_depth", "world_up"],
        }

    def set_camera_extrinsics(
        self,
        world_from_camera: Any,
        *,
        camera_name: str | None = None,
        source: str = "camera_extrinsics",
    ) -> dict[str, Any]:
        """Use a primary camera's real world rotation for 6D input mapping.

        This only changes the mapper's coordinate conversion. It does not
        touch MuJoCo state, controllers, buttons, or filtering history.
        """
        self._control_rotation = camera_frame_to_world_rotation(world_from_camera)
        self._control_camera_name = None if camera_name is None else str(camera_name)
        self._control_rotation_source = str(source)
        return self.describe_control_frame()

    def set_camera_from_sim(self, env_or_sim: Any, camera_name: str | None = None) -> dict[str, Any]:
        """Read ``data.cam_xmat`` from a live MuJoCo / robosuite simulation.

        ``TriViewCompositor`` applies its fixed camera pose without advancing
        physics. Call this after that setup (and after every environment
        reset) to prove that the input basis came from actual extrinsics.
        """
        sim = getattr(env_or_sim, "sim", env_or_sim)
        model = getattr(sim, "model", None)
        data = getattr(sim, "data", None)
        if model is None or data is None:
            raise ValueError("simulation must expose model and data for camera-frame control")
        selected_name = camera_name or self.config.tri_view.front_camera_name
        resolver = getattr(model, "camera_name2id", None)
        if not callable(resolver):
            raise ValueError("simulation model does not expose camera_name2id")
        camera_id = int(resolver(selected_name))
        if camera_id < 0:
            raise ValueError(f"primary camera does not exist: {selected_name!r}")
        native_data = getattr(data, "_data", data)
        matrices = getattr(native_data, "cam_xmat", None)
        if matrices is None:
            raise ValueError("simulation data does not expose cam_xmat")
        rotation = np.asarray(matrices[camera_id], dtype=np.float64).reshape(3, 3)
        return self.set_camera_extrinsics(
            rotation,
            camera_name=selected_name,
            source="live_mujoco_cam_xmat",
        )

    def reset(self) -> None:
        """Discard smoothing state, e.g. on stop, reconnect or reset."""
        self._filtered.fill(0.0)
        self._axis_active.fill(False)
        self._neutral_frames = 0
        self._last_diagnostics = {
            "valid": False,
            "calibrated_axes": [0.0] * 6,
            "deadzone_axes": [0.0] * 6,
            "ema_axes": [0.0] * 6,
            "final_osc_delta": [0.0] * 6,
            "axis_active": [False] * 6,
            "neutral_consecutive_frames": 0,
            "neutral_locked": True,
            "reason": "reset_or_invalid",
        }

    def _invalid(self) -> np.ndarray:
        self.reset()
        return np.zeros(6, dtype=np.float64)

    def diagnostics(self) -> dict[str, Any]:
        """Return the most recent conditioning stages for drift evidence."""
        return {key: (list(value) if isinstance(value, list) else value)
                for key, value in self._last_diagnostics.items()}

    @staticmethod
    def _scale_and_clamp(vector: np.ndarray, scale: float, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros(3, dtype=np.float64)
        # Clamp the magnitude before multiplying by the physical scale. This
        # also avoids overflow for extreme but finite raw input/config values.
        return vector * min(scale, maximum / norm)

    def map(self, raw_axes: Sequence[float] | np.ndarray, valid: bool = True) -> np.ndarray:
        """Return ``[dx, dy, dz, rx, ry, rz]`` in metres / axis-angle radians.

        Invalid validity flags, malformed shape/type, NaN and infinity are
        fail-closed: return exact zero and clear filtering history. Startup
        neutral arming and input timestamps belong to the acquisition layer.
        """
        if not isinstance(valid, (bool, np.bool_)) or not valid:
            return self._invalid()
        try:
            raw = np.asarray(raw_axes)
            if raw.shape != (6,) or raw.dtype.kind not in "iuf":
                return self._invalid()
            raw = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError, OverflowError):
            return self._invalid()
        if not np.all(np.isfinite(raw)):
            return self._invalid()

        # Clip before division so even extreme finite raw values cannot overflow.
        normalized = np.clip(raw, -self._raw_ranges, self._raw_ranges) / self._raw_ranges
        magnitude = np.abs(normalized)
        # Use a larger threshold to start a command and a smaller one to end
        # its active state. The actual vector is zero inside the start band;
        # the latch prevents noisy crossings from reviving an EMA tail.
        self._axis_active = np.where(
            self._axis_active,
            magnitude > self._release_deadzones,
            magnitude > self._deadzones,
        )
        conditioned = np.sign(normalized) * np.maximum(magnitude - self._deadzones, 0.0) / (1.0 - self._deadzones)

        alpha = self.config.smoothing_alpha
        if alpha is None:
            self._filtered[:] = conditioned
        else:
            self._filtered[:] = alpha * conditioned + (1.0 - alpha) * self._filtered
        # Release wins over EMA: a conditioned-zero axis never retains an
        # old tail. After consecutive complete-neutral frames, lock every
        # channel to exact zero rather than slowly decaying toward it.
        self._filtered[np.abs(conditioned) <= _EPSILON] = 0.0
        if not np.any(self._axis_active):
            self._neutral_frames += 1
        else:
            self._neutral_frames = 0
        if self._neutral_frames >= self.config.neutral_lock_frames:
            self._filtered.fill(0.0)
            self._axis_active.fill(False)
        mapped = self._filtered[self._axis_order] * self._axis_inversion
        translation = self._control_rotation @ mapped[:3]
        rotation = self._control_rotation @ mapped[3:]
        output = np.concatenate(
            (
                self._scale_and_clamp(translation, self.config.translation_scale_m, self.config.max_translation_delta_m),
                self._scale_and_clamp(rotation, self.config.rotation_scale_rad, self.config.max_rotation_delta_rad),
            )
        )
        self._last_diagnostics = {
            "valid": True,
            "calibrated_axes": raw.tolist(),
            "normalized_axes": normalized.tolist(),
            "deadzone_axes": conditioned.tolist(),
            "ema_axes": self._filtered.tolist(),
            "final_osc_delta": output.tolist(),
            "axis_active": self._axis_active.tolist(),
            "neutral_consecutive_frames": int(self._neutral_frames),
            "neutral_locked": bool(self._neutral_frames >= self.config.neutral_lock_frames),
            "enter_deadzone": self._deadzones.tolist(),
            "release_deadzone": self._release_deadzones.tolist(),
        }
        return output
