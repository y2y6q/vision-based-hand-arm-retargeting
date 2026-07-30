from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# OpenCV 摄像头坐标：
# +X：画面向右
# +Y：画面向下
# +Z：朝摄像头前方
#
# Panda 基座坐标映射：
# 摄像头 +Z -> Panda +X
# 摄像头 +X -> Panda -Y
# 摄像头 +Y -> Panda -Z
DEFAULT_CAMERA_TO_ROBOT_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# 在机器人 X/Y/Z 轴上的初始比例。
# 后续通过 test_pose_mapper.py 调整。
DEFAULT_POSITION_SCALE = np.array(
    [
        0.030,
        0.045,
        0.045,
    ],
    dtype=np.float64,
)


@dataclass
class MappedPose:
    valid: bool
    position: np.ndarray
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    camera_delta_position: np.ndarray
    robot_delta_position: np.ndarray
    camera_delta_rotation: np.ndarray
    robot_delta_rotation: np.ndarray
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
            "camera_delta_position": (
                self.camera_delta_position.copy()
            ),
            "robot_delta_position": (
                self.robot_delta_position.copy()
            ),
            "camera_delta_rotation": (
                self.camera_delta_rotation.copy()
            ),
            "robot_delta_rotation": (
                self.robot_delta_rotation.copy()
            ),
            "message": self.message,
        }


def project_to_rotation_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(3, 3)

    u_matrix, _, vt_matrix = np.linalg.svd(
        matrix
    )

    rotation = u_matrix @ vt_matrix

    if np.linalg.det(rotation) < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix

    return rotation


def rotation_matrix_to_quaternion_xyzw(
    rotation: np.ndarray,
) -> np.ndarray:
    rotation = project_to_rotation_matrix(
        rotation
    )

    trace = float(
        np.trace(rotation)
    )

    if trace > 0.0:
        scale = np.sqrt(
            trace + 1.0
        ) * 2.0

        qw = 0.25 * scale

        qx = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / scale

        qy = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / scale

        qz = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / scale

    elif (
        rotation[0, 0] > rotation[1, 1]
        and rotation[0, 0] > rotation[2, 2]
    ):
        scale = np.sqrt(
            1.0
            + rotation[0, 0]
            - rotation[1, 1]
            - rotation[2, 2]
        ) * 2.0

        qw = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / scale

        qx = 0.25 * scale

        qy = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / scale

        qz = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / scale

    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(
            1.0
            + rotation[1, 1]
            - rotation[0, 0]
            - rotation[2, 2]
        ) * 2.0

        qw = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / scale

        qx = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / scale

        qy = 0.25 * scale

        qz = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / scale

    else:
        scale = np.sqrt(
            1.0
            + rotation[2, 2]
            - rotation[0, 0]
            - rotation[1, 1]
        ) * 2.0

        qw = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / scale

        qx = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / scale

        qy = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / scale

        qz = 0.25 * scale

    quaternion = np.array(
        [
            qx,
            qy,
            qz,
            qw,
        ],
        dtype=np.float64,
    )

    quaternion_norm = np.linalg.norm(
        quaternion
    )

    if quaternion_norm < 1e-8:
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

    return quaternion / quaternion_norm


def quaternion_xyzw_to_rotation_matrix(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = np.asarray(
        quaternion,
        dtype=np.float64,
    ).reshape(4)

    quaternion_norm = np.linalg.norm(
        quaternion
    )

    if quaternion_norm < 1e-8:
        return np.eye(
            3,
            dtype=np.float64,
        )

    x, y, z, w = (
        quaternion
        / quaternion_norm
    )

    rotation = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )

    return project_to_rotation_matrix(
        rotation
    )


class PoseMapper:
    def __init__(
        self,
        camera_to_robot_rotation: Optional[
            np.ndarray
        ] = None,
        position_scale: Optional[
            np.ndarray
        ] = None,
    ) -> None:
        if camera_to_robot_rotation is None:
            camera_to_robot_rotation = (
                DEFAULT_CAMERA_TO_ROBOT_ROTATION
            )

        if position_scale is None:
            position_scale = (
                DEFAULT_POSITION_SCALE
            )

        self.camera_to_robot_rotation = (
            project_to_rotation_matrix(
                camera_to_robot_rotation
            )
        )

        self.position_scale = np.asarray(
            position_scale,
            dtype=np.float64,
        ).reshape(3)

        self.camera_reference_position: Optional[
            np.ndarray
        ] = None

        self.camera_reference_rotation: Optional[
            np.ndarray
        ] = None

        self.robot_reference_position: Optional[
            np.ndarray
        ] = None

        self.robot_reference_rotation: Optional[
            np.ndarray
        ] = None

    @property
    def is_calibrated(
        self,
    ) -> bool:
        return (
            self.camera_reference_position
            is not None
            and self.camera_reference_rotation
            is not None
            and self.robot_reference_position
            is not None
            and self.robot_reference_rotation
            is not None
        )

    def set_reference(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        robot_position: np.ndarray,
        robot_rotation: np.ndarray,
    ) -> None:
        self.camera_reference_position = (
            np.asarray(
                camera_position,
                dtype=np.float64,
            ).reshape(3)
        )

        self.camera_reference_rotation = (
            project_to_rotation_matrix(
                camera_rotation
            )
        )

        self.robot_reference_position = (
            np.asarray(
                robot_position,
                dtype=np.float64,
            ).reshape(3)
        )

        robot_rotation_array = np.asarray(
            robot_rotation,
            dtype=np.float64,
        )

        if robot_rotation_array.size == 4:
            self.robot_reference_rotation = (
                quaternion_xyzw_to_rotation_matrix(
                    robot_rotation_array
                )
            )
        else:
            self.robot_reference_rotation = (
                project_to_rotation_matrix(
                    robot_rotation_array.reshape(
                        3,
                        3,
                    )
                )
            )

    def clear_reference(
        self,
    ) -> None:
        self.camera_reference_position = None
        self.camera_reference_rotation = None
        self.robot_reference_position = None
        self.robot_reference_rotation = None

    def set_position_scale(
        self,
        position_scale: np.ndarray,
    ) -> None:
        self.position_scale = np.asarray(
            position_scale,
            dtype=np.float64,
        ).reshape(3)

    def set_camera_to_robot_rotation(
        self,
        rotation: np.ndarray,
    ) -> None:
        self.camera_to_robot_rotation = (
            project_to_rotation_matrix(
                rotation
            )
        )

    def map_pose(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
    ) -> MappedPose:
        identity_rotation = np.eye(
            3,
            dtype=np.float64,
        )

        zero_vector = np.zeros(
            3,
            dtype=np.float64,
        )

        identity_quaternion = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        if not self.is_calibrated:
            return MappedPose(
                valid=False,
                position=zero_vector,
                rotation_matrix=identity_rotation,
                quaternion_xyzw=identity_quaternion,
                camera_delta_position=zero_vector,
                robot_delta_position=zero_vector,
                camera_delta_rotation=identity_rotation,
                robot_delta_rotation=identity_rotation,
                message="Pose mapper is not calibrated.",
            )

        current_camera_position = (
            np.asarray(
                camera_position,
                dtype=np.float64,
            ).reshape(3)
        )

        current_camera_rotation = (
            project_to_rotation_matrix(
                camera_rotation
            )
        )

        camera_delta_position = (
            current_camera_position
            - self.camera_reference_position
        )

        unscaled_robot_delta = (
            self.camera_to_robot_rotation
            @ camera_delta_position
        )

        robot_delta_position = (
            self.position_scale
            * unscaled_robot_delta
        )

        target_robot_position = (
            self.robot_reference_position
            + robot_delta_position
        )

        camera_delta_rotation = (
            current_camera_rotation
            @ self.camera_reference_rotation.T
        )

        camera_delta_rotation = (
            project_to_rotation_matrix(
                camera_delta_rotation
            )
        )

        robot_delta_rotation = (
            self.camera_to_robot_rotation
            @ camera_delta_rotation
            @ self.camera_to_robot_rotation.T
        )

        robot_delta_rotation = (
            project_to_rotation_matrix(
                robot_delta_rotation
            )
        )

        target_robot_rotation = (
            robot_delta_rotation
            @ self.robot_reference_rotation
        )

        target_robot_rotation = (
            project_to_rotation_matrix(
                target_robot_rotation
            )
        )

        target_quaternion = (
            rotation_matrix_to_quaternion_xyzw(
                target_robot_rotation
            )
        )

        valid = (
            np.all(
                np.isfinite(
                    target_robot_position
                )
            )
            and np.all(
                np.isfinite(
                    target_robot_rotation
                )
            )
            and np.all(
                np.isfinite(
                    target_quaternion
                )
            )
        )

        return MappedPose(
            valid=valid,
            position=target_robot_position,
            rotation_matrix=target_robot_rotation,
            quaternion_xyzw=target_quaternion,
            camera_delta_position=(
                camera_delta_position
            ),
            robot_delta_position=(
                robot_delta_position
            ),
            camera_delta_rotation=(
                camera_delta_rotation
            ),
            robot_delta_rotation=(
                robot_delta_rotation
            ),
            message=(
                "Pose mapped."
                if valid
                else "Mapped pose contains invalid values."
            ),
        )

    def save_configuration(
        self,
        file_path: str,
    ) -> None:
        output_path = Path(
            file_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            str(output_path),
            camera_to_robot_rotation=(
                self.camera_to_robot_rotation
            ),
            position_scale=(
                self.position_scale
            ),
        )

    def load_configuration(
        self,
        file_path: str,
    ) -> None:
        input_path = Path(
            file_path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Mapping configuration does not exist: "
                f"{input_path}"
            )

        data = np.load(
            str(input_path)
        )

        self.camera_to_robot_rotation = (
            project_to_rotation_matrix(
                data[
                    "camera_to_robot_rotation"
                ]
            )
        )

        self.position_scale = np.asarray(
            data["position_scale"],
            dtype=np.float64,
        ).reshape(3)