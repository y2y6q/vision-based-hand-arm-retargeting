from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.teleop.pose_mapper import (
    project_to_rotation_matrix,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_quaternion_xyzw,
)


@dataclass
class FilteredPose:
    valid: bool
    position: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    position_clipped: bool
    rotation_clipped: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "position": self.position.copy(),
            "rotation_matrix": (
                self.rotation_matrix.copy()
            ),
            "quaternion_xyzw": (
                self.quaternion_xyzw.copy()
            ),
            "linear_velocity": (
                self.linear_velocity.copy()
            ),
            "angular_velocity": (
                self.angular_velocity.copy()
            ),
            "position_clipped": bool(
                self.position_clipped
            ),
            "rotation_clipped": bool(
                self.rotation_clipped
            ),
            "message": self.message,
        }


def clip_vector_norm(
    vector: np.ndarray,
    maximum_norm: float,
) -> Tuple[np.ndarray, bool]:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    ).reshape(3)

    vector_norm = float(
        np.linalg.norm(vector)
    )

    if (
        maximum_norm <= 0.0
        or vector_norm <= maximum_norm
        or vector_norm < 1e-10
    ):
        return vector.copy(), False

    return (
        vector
        * maximum_norm
        / vector_norm,
        True,
    )


def rotation_matrix_to_rotvec(
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    rotation = project_to_rotation_matrix(
        rotation_matrix
    )

    cosine_angle = float(
        np.clip(
            (
                np.trace(rotation)
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    angle = math.acos(
        cosine_angle
    )

    if angle < 1e-8:
        return np.zeros(
            3,
            dtype=np.float64,
        )

    if abs(
        math.pi - angle
    ) < 1e-5:
        diagonal = np.diag(
            rotation
        )

        axis = np.sqrt(
            np.maximum(
                (
                    diagonal
                    + 1.0
                )
                / 2.0,
                0.0,
            )
        )

        if (
            rotation[2, 1]
            - rotation[1, 2]
        ) < 0.0:
            axis[0] *= -1.0

        if (
            rotation[0, 2]
            - rotation[2, 0]
        ) < 0.0:
            axis[1] *= -1.0

        if (
            rotation[1, 0]
            - rotation[0, 1]
        ) < 0.0:
            axis[2] *= -1.0

        axis_norm = np.linalg.norm(
            axis
        )

        if axis_norm < 1e-8:
            axis = np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float64,
            )

        else:
            axis = axis / axis_norm

        return axis * angle

    axis = np.array(
        [
            rotation[2, 1]
            - rotation[1, 2],
            rotation[0, 2]
            - rotation[2, 0],
            rotation[1, 0]
            - rotation[0, 1],
        ],
        dtype=np.float64,
    )

    axis /= (
        2.0
        * math.sin(angle)
    )

    return axis * angle


def rotvec_to_rotation_matrix(
    rotation_vector: np.ndarray,
) -> np.ndarray:
    rotation_vector = np.asarray(
        rotation_vector,
        dtype=np.float64,
    ).reshape(3)

    angle = float(
        np.linalg.norm(
            rotation_vector
        )
    )

    if angle < 1e-10:
        return np.eye(
            3,
            dtype=np.float64,
        )

    axis = (
        rotation_vector
        / angle
    )

    x_value, y_value, z_value = axis

    skew_matrix = np.array(
        [
            [
                0.0,
                -z_value,
                y_value,
            ],
            [
                z_value,
                0.0,
                -x_value,
            ],
            [
                -y_value,
                x_value,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    identity = np.eye(
        3,
        dtype=np.float64,
    )

    rotation = (
        identity
        + math.sin(angle)
        * skew_matrix
        + (
            1.0
            - math.cos(angle)
        )
        * (
            skew_matrix
            @ skew_matrix
        )
    )

    return project_to_rotation_matrix(
        rotation
    )


def calculate_alpha(
    cutoff_frequency: float,
    delta_time: float,
) -> float:
    cutoff_frequency = max(
        float(cutoff_frequency),
        1e-6,
    )

    delta_time = max(
        float(delta_time),
        1e-6,
    )

    time_constant = (
        1.0
        / (
            2.0
            * math.pi
            * cutoff_frequency
        )
    )

    return float(
        1.0
        / (
            1.0
            + time_constant
            / delta_time
        )
    )


class LowPassVectorFilter:
    def __init__(
        self,
    ) -> None:
        self.value: Optional[
            np.ndarray
        ] = None

    def reset(
        self,
    ) -> None:
        self.value = None

    def filter(
        self,
        raw_value: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        raw_value = np.asarray(
            raw_value,
            dtype=np.float64,
        ).reshape(3)

        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )

        if self.value is None:
            self.value = (
                raw_value.copy()
            )

        else:
            self.value = (
                alpha
                * raw_value
                + (
                    1.0
                    - alpha
                )
                * self.value
            )

        return self.value.copy()


class OneEuroVectorFilter:
    def __init__(
        self,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
    ) -> None:
        self.min_cutoff = float(
            min_cutoff
        )

        self.beta = float(
            beta
        )

        self.derivative_cutoff = float(
            derivative_cutoff
        )

        self.previous_raw_value: Optional[
            np.ndarray
        ] = None

        self.value_filter = (
            LowPassVectorFilter()
        )

        self.derivative_filter = (
            LowPassVectorFilter()
        )

    def reset(
        self,
    ) -> None:
        self.previous_raw_value = None
        self.value_filter.reset()
        self.derivative_filter.reset()

    def filter(
        self,
        raw_value: np.ndarray,
        delta_time: float,
    ) -> np.ndarray:
        raw_value = np.asarray(
            raw_value,
            dtype=np.float64,
        ).reshape(3)

        delta_time = max(
            float(delta_time),
            1e-6,
        )

        if self.previous_raw_value is None:
            raw_derivative = np.zeros(
                3,
                dtype=np.float64,
            )

        else:
            raw_derivative = (
                raw_value
                - self.previous_raw_value
            ) / delta_time

        derivative_alpha = (
            calculate_alpha(
                self.derivative_cutoff,
                delta_time,
            )
        )

        filtered_derivative = (
            self.derivative_filter.filter(
                raw_derivative,
                derivative_alpha,
            )
        )

        adaptive_cutoff = (
            self.min_cutoff
            + self.beta
            * float(
                np.linalg.norm(
                    filtered_derivative
                )
            )
        )

        value_alpha = calculate_alpha(
            adaptive_cutoff,
            delta_time,
        )

        filtered_value = (
            self.value_filter.filter(
                raw_value,
                value_alpha,
            )
        )

        self.previous_raw_value = (
            raw_value.copy()
        )

        return filtered_value


class OneEuroRotationFilter:
    def __init__(
        self,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
    ) -> None:
        self.min_cutoff = float(
            min_cutoff
        )

        self.beta = float(
            beta
        )

        self.derivative_cutoff = float(
            derivative_cutoff
        )

        self.previous_raw_rotation: Optional[
            np.ndarray
        ] = None

        self.filtered_rotation: Optional[
            np.ndarray
        ] = None

        self.derivative_filter = (
            LowPassVectorFilter()
        )

    def reset(
        self,
    ) -> None:
        self.previous_raw_rotation = None
        self.filtered_rotation = None
        self.derivative_filter.reset()

    def filter(
        self,
        raw_rotation: np.ndarray,
        delta_time: float,
    ) -> np.ndarray:
        raw_rotation = (
            project_to_rotation_matrix(
                raw_rotation
            )
        )

        delta_time = max(
            float(delta_time),
            1e-6,
        )

        if (
            self.previous_raw_rotation
            is None
            or self.filtered_rotation
            is None
        ):
            self.previous_raw_rotation = (
                raw_rotation.copy()
            )

            self.filtered_rotation = (
                raw_rotation.copy()
            )

            return (
                self.filtered_rotation.copy()
            )

        raw_delta_rotation = (
            raw_rotation
            @ self.previous_raw_rotation.T
        )

        raw_angular_velocity = (
            rotation_matrix_to_rotvec(
                raw_delta_rotation
            )
            / delta_time
        )

        derivative_alpha = (
            calculate_alpha(
                self.derivative_cutoff,
                delta_time,
            )
        )

        filtered_angular_velocity = (
            self.derivative_filter.filter(
                raw_angular_velocity,
                derivative_alpha,
            )
        )

        adaptive_cutoff = (
            self.min_cutoff
            + self.beta
            * float(
                np.linalg.norm(
                    filtered_angular_velocity
                )
            )
        )

        rotation_alpha = calculate_alpha(
            adaptive_cutoff,
            delta_time,
        )

        filter_delta_rotation = (
            raw_rotation
            @ self.filtered_rotation.T
        )

        filter_delta_rotvec = (
            rotation_matrix_to_rotvec(
                filter_delta_rotation
            )
        )

        applied_rotation = (
            rotvec_to_rotation_matrix(
                rotation_alpha
                * filter_delta_rotvec
            )
        )

        self.filtered_rotation = (
            applied_rotation
            @ self.filtered_rotation
        )

        self.filtered_rotation = (
            project_to_rotation_matrix(
                self.filtered_rotation
            )
        )

        self.previous_raw_rotation = (
            raw_rotation.copy()
        )

        return (
            self.filtered_rotation.copy()
        )


class PoseFilter:
    def __init__(
        self,
        position_min_cutoff: float = 1.0,
        position_beta: float = 0.03,
        position_derivative_cutoff: float = 1.0,
        rotation_min_cutoff: float = 1.5,
        rotation_beta: float = 0.05,
        rotation_derivative_cutoff: float = 1.0,
        maximum_linear_velocity: float = 0.20,
        maximum_linear_acceleration: float = 0.80,
        maximum_angular_velocity: float = 1.20,
        maximum_angular_acceleration: float = 4.00,
        workspace_half_extent: Optional[
            np.ndarray
        ] = None,
        minimum_z: Optional[float] = None,
        maximum_relative_rotation_degrees: float = 120.0,
    ) -> None:
        if workspace_half_extent is None:
            workspace_half_extent = (
                np.array(
                    [
                        0.25,
                        0.25,
                        0.25,
                    ],
                    dtype=np.float64,
                )
            )

        self.maximum_linear_velocity = float(
            maximum_linear_velocity
        )

        self.maximum_linear_acceleration = float(
            maximum_linear_acceleration
        )

        self.maximum_angular_velocity = float(
            maximum_angular_velocity
        )

        self.maximum_angular_acceleration = float(
            maximum_angular_acceleration
        )

        self.workspace_half_extent = (
            np.asarray(
                workspace_half_extent,
                dtype=np.float64,
            ).reshape(3)
        )

        self.minimum_z = (
            None
            if minimum_z is None
            else float(minimum_z)
        )

        self.maximum_relative_rotation = (
            math.radians(
                float(
                    maximum_relative_rotation_degrees
                )
            )
        )

        self.position_filter = (
            OneEuroVectorFilter(
                min_cutoff=(
                    position_min_cutoff
                ),
                beta=position_beta,
                derivative_cutoff=(
                    position_derivative_cutoff
                ),
            )
        )

        self.rotation_filter = (
            OneEuroRotationFilter(
                min_cutoff=(
                    rotation_min_cutoff
                ),
                beta=rotation_beta,
                derivative_cutoff=(
                    rotation_derivative_cutoff
                ),
            )
        )

        self.reference_position: Optional[
            np.ndarray
        ] = None

        self.reference_rotation: Optional[
            np.ndarray
        ] = None

        self.previous_position: Optional[
            np.ndarray
        ] = None

        self.previous_rotation: Optional[
            np.ndarray
        ] = None

        self.previous_linear_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.previous_angular_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.previous_timestamp: Optional[
            float
        ] = None

    @property
    def is_initialized(
        self,
    ) -> bool:
        return (
            self.reference_position
            is not None
            and self.reference_rotation
            is not None
            and self.previous_position
            is not None
            and self.previous_rotation
            is not None
        )

    def reset(
        self,
    ) -> None:
        self.position_filter.reset()
        self.rotation_filter.reset()

        self.reference_position = None
        self.reference_rotation = None

        self.previous_position = None
        self.previous_rotation = None

        self.previous_linear_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.previous_angular_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.previous_timestamp = None

    def set_reference(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        timestamp: Optional[
            float
        ] = None,
    ) -> None:
        position = np.asarray(
            position,
            dtype=np.float64,
        ).reshape(3)

        rotation_array = np.asarray(
            rotation,
            dtype=np.float64,
        )

        if rotation_array.size == 4:
            rotation_matrix = (
                quaternion_xyzw_to_rotation_matrix(
                    rotation_array
                )
            )

        else:
            rotation_matrix = (
                project_to_rotation_matrix(
                    rotation_array.reshape(
                        3,
                        3,
                    )
                )
            )

        self.reset()

        self.reference_position = (
            position.copy()
        )

        self.reference_rotation = (
            rotation_matrix.copy()
        )

        self.previous_position = (
            position.copy()
        )

        self.previous_rotation = (
            rotation_matrix.copy()
        )

        current_time = (
            time.monotonic()
            if timestamp is None
            else float(timestamp)
        )

        self.previous_timestamp = (
            current_time
        )

        self.position_filter.filter(
            position,
            1.0 / 30.0,
        )

        self.rotation_filter.filter(
            rotation_matrix,
            1.0 / 30.0,
        )

    def _hold_previous_pose(
        self,
        message: str,
    ) -> FilteredPose:
        if not self.is_initialized:
            return FilteredPose(
                valid=False,
                position=np.zeros(
                    3,
                    dtype=np.float64,
                ),
                rotation_matrix=np.eye(
                    3,
                    dtype=np.float64,
                ),
                quaternion_xyzw=np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
                linear_velocity=np.zeros(
                    3,
                    dtype=np.float64,
                ),
                angular_velocity=np.zeros(
                    3,
                    dtype=np.float64,
                ),
                position_clipped=False,
                rotation_clipped=False,
                message=message,
            )

        return FilteredPose(
            valid=False,
            position=(
                self.previous_position.copy()
            ),
            rotation_matrix=(
                self.previous_rotation.copy()
            ),
            quaternion_xyzw=(
                rotation_matrix_to_quaternion_xyzw(
                    self.previous_rotation
                )
            ),
            linear_velocity=(
                self.previous_linear_velocity.copy()
            ),
            angular_velocity=(
                self.previous_angular_velocity.copy()
            ),
            position_clipped=False,
            rotation_clipped=False,
            message=message,
        )

    def filter_pose(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        timestamp: Optional[
            float
        ] = None,
        input_valid: bool = True,
    ) -> FilteredPose:
        if not input_valid:
            return self._hold_previous_pose(
                "Input pose rejected; holding previous pose."
            )

        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        ).reshape(3)

        rotation_array = np.asarray(
            target_rotation,
            dtype=np.float64,
        )

        if rotation_array.size == 4:
            target_rotation_matrix = (
                quaternion_xyzw_to_rotation_matrix(
                    rotation_array
                )
            )

        else:
            target_rotation_matrix = (
                project_to_rotation_matrix(
                    rotation_array.reshape(
                        3,
                        3,
                    )
                )
            )

        if (
            not np.all(
                np.isfinite(
                    target_position
                )
            )
            or not np.all(
                np.isfinite(
                    target_rotation_matrix
                )
            )
        ):
            return self._hold_previous_pose(
                "Input pose contains invalid values."
            )

        current_time = (
            time.monotonic()
            if timestamp is None
            else float(timestamp)
        )

        if not self.is_initialized:
            self.set_reference(
                target_position,
                target_rotation_matrix,
                timestamp=current_time,
            )

            return FilteredPose(
                valid=True,
                position=target_position.copy(),
                rotation_matrix=(
                    target_rotation_matrix.copy()
                ),
                quaternion_xyzw=(
                    rotation_matrix_to_quaternion_xyzw(
                        target_rotation_matrix
                    )
                ),
                linear_velocity=np.zeros(
                    3,
                    dtype=np.float64,
                ),
                angular_velocity=np.zeros(
                    3,
                    dtype=np.float64,
                ),
                position_clipped=False,
                rotation_clipped=False,
                message="Pose filter initialized.",
            )

        delta_time = (
            current_time
            - self.previous_timestamp
        )

        delta_time = float(
            np.clip(
                delta_time,
                1.0 / 240.0,
                0.10,
            )
        )

        minimum_position = (
            self.reference_position
            - self.workspace_half_extent
        )

        maximum_position = (
            self.reference_position
            + self.workspace_half_extent
        )

        workspace_position = np.clip(
            target_position,
            minimum_position,
            maximum_position,
        )

        if self.minimum_z is not None:
            workspace_position[2] = max(
                workspace_position[2],
                self.minimum_z,
            )

        position_clipped = not np.allclose(
            workspace_position,
            target_position,
            atol=1e-9,
        )

        relative_rotation = (
            target_rotation_matrix
            @ self.reference_rotation.T
        )

        relative_rotvec = (
            rotation_matrix_to_rotvec(
                relative_rotation
            )
        )

        relative_angle = float(
            np.linalg.norm(
                relative_rotvec
            )
        )

        rotation_clipped = False

        if (
            relative_angle
            > self.maximum_relative_rotation
            and relative_angle > 1e-8
        ):
            relative_rotvec = (
                relative_rotvec
                * self.maximum_relative_rotation
                / relative_angle
            )

            target_rotation_matrix = (
                rotvec_to_rotation_matrix(
                    relative_rotvec
                )
                @ self.reference_rotation
            )

            target_rotation_matrix = (
                project_to_rotation_matrix(
                    target_rotation_matrix
                )
            )

            rotation_clipped = True

        filtered_position = (
            self.position_filter.filter(
                workspace_position,
                delta_time,
            )
        )

        filtered_rotation = (
            self.rotation_filter.filter(
                target_rotation_matrix,
                delta_time,
            )
        )

        desired_linear_velocity = (
            filtered_position
            - self.previous_position
        ) / delta_time

        linear_acceleration = (
            desired_linear_velocity
            - self.previous_linear_velocity
        ) / delta_time

        limited_linear_acceleration, acceleration_clipped = (
            clip_vector_norm(
                linear_acceleration,
                self.maximum_linear_acceleration,
            )
        )

        limited_linear_velocity = (
            self.previous_linear_velocity
            + limited_linear_acceleration
            * delta_time
        )

        limited_linear_velocity, velocity_clipped = (
            clip_vector_norm(
                limited_linear_velocity,
                self.maximum_linear_velocity,
            )
        )

        output_position = (
            self.previous_position
            + limited_linear_velocity
            * delta_time
        )

        desired_delta_rotation = (
            filtered_rotation
            @ self.previous_rotation.T
        )

        desired_angular_velocity = (
            rotation_matrix_to_rotvec(
                desired_delta_rotation
            )
            / delta_time
        )

        angular_acceleration = (
            desired_angular_velocity
            - self.previous_angular_velocity
        ) / delta_time

        limited_angular_acceleration, angular_acceleration_clipped = (
            clip_vector_norm(
                angular_acceleration,
                self.maximum_angular_acceleration,
            )
        )

        limited_angular_velocity = (
            self.previous_angular_velocity
            + limited_angular_acceleration
            * delta_time
        )

        limited_angular_velocity, angular_velocity_clipped = (
            clip_vector_norm(
                limited_angular_velocity,
                self.maximum_angular_velocity,
            )
        )

        output_rotation = (
            rotvec_to_rotation_matrix(
                limited_angular_velocity
                * delta_time
            )
            @ self.previous_rotation
        )

        output_rotation = (
            project_to_rotation_matrix(
                output_rotation
            )
        )

        position_clipped = (
            position_clipped
            or acceleration_clipped
            or velocity_clipped
        )

        rotation_clipped = (
            rotation_clipped
            or angular_acceleration_clipped
            or angular_velocity_clipped
        )

        self.previous_position = (
            output_position.copy()
        )

        self.previous_rotation = (
            output_rotation.copy()
        )

        self.previous_linear_velocity = (
            limited_linear_velocity.copy()
        )

        self.previous_angular_velocity = (
            limited_angular_velocity.copy()
        )

        self.previous_timestamp = (
            current_time
        )

        return FilteredPose(
            valid=True,
            position=output_position,
            rotation_matrix=output_rotation,
            quaternion_xyzw=(
                rotation_matrix_to_quaternion_xyzw(
                    output_rotation
                )
            ),
            linear_velocity=(
                limited_linear_velocity
            ),
            angular_velocity=(
                limited_angular_velocity
            ),
            position_clipped=(
                position_clipped
            ),
            rotation_clipped=(
                rotation_clipped
            ),
            message="Pose filtered and constrained.",
        )