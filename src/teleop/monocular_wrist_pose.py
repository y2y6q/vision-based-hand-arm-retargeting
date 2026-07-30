from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PALM_LANDMARK_IDS = (
    0,   # wrist
    1,   # thumb_cmc
    5,   # index_mcp
    9,   # middle_mcp
    13,  # ring_mcp
    17,  # pinky_mcp
)


@dataclass
class WristPoseEstimate:
    valid: bool
    rotation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    translation: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error: float
    inlier_count: int
    pnp_weight: float
    scale_depth: float
    hand_scale: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "rotation_matrix": self.rotation_matrix.copy(),
            "quaternion_xyzw": self.quaternion_xyzw.copy(),
            "translation": self.translation.copy(),
            "rvec": self.rvec.copy(),
            "tvec": self.tvec.copy(),
            "reprojection_error": float(self.reprojection_error),
            "inlier_count": int(self.inlier_count),
            "pnp_weight": float(self.pnp_weight),
            "scale_depth": float(self.scale_depth),
            "hand_scale": float(self.hand_scale),
            "message": self.message,
        }


def normalize_vector(
    vector: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = np.linalg.norm(vector)

    if norm < epsilon:
        return np.zeros_like(vector)

    return vector / norm


def rotation_matrix_to_quaternion_xyzw(
    rotation: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    )

    trace = np.trace(rotation)

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

    quaternion /= (
        np.linalg.norm(quaternion)
        + 1e-8
    )

    return quaternion


def quaternion_xyzw_to_rotation_matrix(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = np.asarray(
        quaternion,
        dtype=np.float64,
    )

    quaternion /= (
        np.linalg.norm(quaternion)
        + 1e-8
    )

    x, y, z, w = quaternion

    return np.array(
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


def slerp_quaternion_xyzw(
    first: np.ndarray,
    second: np.ndarray,
    second_weight: float,
) -> np.ndarray:
    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    first /= (
        np.linalg.norm(first)
        + 1e-8
    )

    second /= (
        np.linalg.norm(second)
        + 1e-8
    )

    second_weight = float(
        np.clip(
            second_weight,
            0.0,
            1.0,
        )
    )

    dot = float(
        np.dot(
            first,
            second,
        )
    )

    if dot < 0.0:
        second = -second
        dot = -dot

    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )

    if dot > 0.9995:
        result = (
            first
            + second_weight
            * (second - first)
        )

        result /= (
            np.linalg.norm(result)
            + 1e-8
        )

        return result

    theta_0 = np.arccos(dot)
    theta = theta_0 * second_weight

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    first_scale = (
        np.cos(theta)
        - dot
        * sin_theta
        / sin_theta_0
    )

    second_scale = (
        sin_theta
        / sin_theta_0
    )

    result = (
        first_scale * first
        + second_scale * second
    )

    result /= (
        np.linalg.norm(result)
        + 1e-8
    )

    return result


def landmark_to_array(
    landmark: Any,
) -> np.ndarray:
    return np.array(
        [
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
        ],
        dtype=np.float64,
    )


def extract_landmarks_xyz(
    landmarks: Sequence[Any],
) -> np.ndarray:
    return np.asarray(
        [
            landmark_to_array(
                landmark
            )
            for landmark
            in landmarks
        ],
        dtype=np.float64,
    )


def extract_palm_points(
    landmarks: Sequence[Any],
) -> np.ndarray:
    all_points = extract_landmarks_xyz(
        landmarks
    )

    return all_points[
        list(PALM_LANDMARK_IDS)
    ]


def normalized_points_to_pixels(
    landmarks: Sequence[Any],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = []

    for landmark_id in PALM_LANDMARK_IDS:
        landmark = landmarks[
            landmark_id
        ]

        points.append(
            [
                float(landmark.x)
                * float(image_width),
                float(landmark.y)
                * float(image_height),
            ]
        )

    return np.asarray(
        points,
        dtype=np.float64,
    )


def estimate_hand_scale(
    landmarks: Sequence[Any],
) -> float:
    points = extract_landmarks_xyz(
        landmarks
    )

    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]

    palm_width = np.linalg.norm(
        index_mcp - pinky_mcp
    )

    palm_length = np.linalg.norm(
        middle_mcp - wrist
    )

    return float(
        0.60 * palm_width
        + 0.40 * palm_length
    )


def estimate_landmark_palm_frame(
    landmarks: Sequence[Any],
) -> np.ndarray:
    points = extract_landmarks_xyz(
        landmarks
    )

    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]

    x_axis = normalize_vector(
        index_mcp - pinky_mcp
    )

    y_seed = normalize_vector(
        middle_mcp - wrist
    )

    z_axis = normalize_vector(
        np.cross(
            x_axis,
            y_seed,
        )
    )

    y_axis = normalize_vector(
        np.cross(
            z_axis,
            x_axis,
        )
    )

    rotation = np.column_stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ]
    )

    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0

    return rotation.astype(
        np.float64
    )


class MonocularWristPoseEstimator:
    def __init__(
        self,
        intrinsics_path: str,
        calibration_frame_count: int = 45,
        maximum_reprojection_error: float = 10.0,
    ) -> None:
        self.intrinsics_path = Path(
            intrinsics_path
        )

        self.calibration_frame_count = int(
            calibration_frame_count
        )

        self.maximum_reprojection_error = float(
            maximum_reprojection_error
        )

        self.camera_matrix: np.ndarray
        self.dist_coeffs: np.ndarray
        self.image_width: int
        self.image_height: int

        self._load_intrinsics()

        self._template_samples: List[
            np.ndarray
        ] = []

        self._scale_samples: List[
            float
        ] = []

        self.palm_template_3d: Optional[
            np.ndarray
        ] = None

        self.reference_hand_scale: Optional[
            float
        ] = None

        self.reference_pnp_depth: Optional[
            float
        ] = None

        self.landmark_alignment: Optional[
            np.ndarray
        ] = None

        self.previous_rvec: Optional[
            np.ndarray
        ] = None

        self.previous_tvec: Optional[
            np.ndarray
        ] = None

    def _load_intrinsics(
        self,
    ) -> None:
        if not self.intrinsics_path.exists():
            raise FileNotFoundError(
                "Camera intrinsics file does not exist: "
                f"{self.intrinsics_path}"
            )

        data = np.load(
            str(self.intrinsics_path)
        )

        self.camera_matrix = np.asarray(
            data["camera_matrix"],
            dtype=np.float64,
        )

        self.dist_coeffs = np.asarray(
            data["dist_coeffs"],
            dtype=np.float64,
        )

        self.image_width = int(
            data["image_width"]
        )

        self.image_height = int(
            data["image_height"]
        )

    @property
    def is_calibrated(
        self,
    ) -> bool:
        return (
            self.palm_template_3d
            is not None
            and self.reference_hand_scale
            is not None
        )

    @property
    def calibration_progress(
        self,
    ) -> Tuple[int, int]:
        return (
            len(self._template_samples),
            self.calibration_frame_count,
        )

    def reset_calibration(
        self,
    ) -> None:
        self._template_samples = []
        self._scale_samples = []

        self.palm_template_3d = None
        self.reference_hand_scale = None
        self.reference_pnp_depth = None
        self.landmark_alignment = None

        self.previous_rvec = None
        self.previous_tvec = None

    def add_calibration_frame(
        self,
        normalized_landmarks: Sequence[Any],
        world_landmarks: Optional[
            Sequence[Any]
        ] = None,
    ) -> bool:
        if world_landmarks is not None:
            palm_points = extract_palm_points(
                world_landmarks
            )

        else:
            palm_points = extract_palm_points(
                normalized_landmarks
            )

        palm_points = (
            palm_points
            - palm_points[0]
        )

        palm_scale = np.linalg.norm(
            palm_points[2]
            - palm_points[5]
        )

        if palm_scale < 1e-7:
            return False

        normalized_template = (
            palm_points
            / palm_scale
        )

        self._template_samples.append(
            normalized_template
        )

        self._scale_samples.append(
            estimate_hand_scale(
                normalized_landmarks
            )
        )

        if (
            len(self._template_samples)
            < self.calibration_frame_count
        ):
            return False

        stacked_templates = np.stack(
            self._template_samples,
            axis=0,
        )

        self.palm_template_3d = np.median(
            stacked_templates,
            axis=0,
        ).astype(
            np.float64
        )

        self.reference_hand_scale = float(
            np.median(
                np.asarray(
                    self._scale_samples,
                    dtype=np.float64,
                )
            )
        )

        self.previous_rvec = None
        self.previous_tvec = None
        self.reference_pnp_depth = None
        self.landmark_alignment = None

        return True

    def _calculate_reprojection_error(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> float:
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )

        projected_points = projected_points.reshape(
            -1,
            2,
        )

        error = np.linalg.norm(
            projected_points
            - image_points,
            axis=1,
        )

        return float(
            np.mean(error)
        )

    def _solve_pnp(
        self,
        image_points: np.ndarray,
    ) -> Tuple[
        bool,
        np.ndarray,
        np.ndarray,
        int,
        float,
    ]:
        if self.palm_template_3d is None:
            return (
                False,
                np.zeros(
                    (3, 1),
                    dtype=np.float64,
                ),
                np.zeros(
                    (3, 1),
                    dtype=np.float64,
                ),
                0,
                float("inf"),
            )

        object_points = np.asarray(
            self.palm_template_3d,
            dtype=np.float64,
        )

        success, rvec, tvec, inliers = (
            cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                iterationsCount=120,
                reprojectionError=7.0,
                confidence=0.995,
                flags=cv2.SOLVEPNP_EPNP,
            )
        )

        if not success:
            return (
                False,
                np.zeros(
                    (3, 1),
                    dtype=np.float64,
                ),
                np.zeros(
                    (3, 1),
                    dtype=np.float64,
                ),
                0,
                float("inf"),
            )

        try:
            if hasattr(
                cv2,
                "solvePnPRefineVVS",
            ):
                rvec, tvec = (
                    cv2.solvePnPRefineVVS(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                    )
                )

            elif hasattr(
                cv2,
                "solvePnPRefineLM",
            ):
                rvec, tvec = (
                    cv2.solvePnPRefineLM(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                    )
                )

        except cv2.error:
            pass

        reprojection_error = (
            self._calculate_reprojection_error(
                object_points=object_points,
                image_points=image_points,
                rvec=rvec,
                tvec=tvec,
            )
        )

        inlier_count = (
            int(len(inliers))
            if inliers is not None
            else 0
        )

        return (
            True,
            np.asarray(
                rvec,
                dtype=np.float64,
            ).reshape(
                3,
                1,
            ),
            np.asarray(
                tvec,
                dtype=np.float64,
            ).reshape(
                3,
                1,
            ),
            inlier_count,
            reprojection_error,
        )

    def estimate(
        self,
        normalized_landmarks: Sequence[Any],
        world_landmarks: Optional[
            Sequence[Any]
        ] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> WristPoseEstimate:
        identity_rotation = np.eye(
            3,
            dtype=np.float64,
        )

        zero_vector = np.zeros(
            3,
            dtype=np.float64,
        )

        zero_column = np.zeros(
            (3, 1),
            dtype=np.float64,
        )

        if not self.is_calibrated:
            return WristPoseEstimate(
                valid=False,
                rotation_matrix=identity_rotation,
                quaternion_xyzw=np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
                translation=zero_vector,
                rvec=zero_column,
                tvec=zero_column,
                reprojection_error=float("inf"),
                inlier_count=0,
                pnp_weight=0.0,
                scale_depth=0.0,
                hand_scale=0.0,
                message="Estimator is not calibrated.",
            )

        width = (
            int(image_width)
            if image_width is not None
            else self.image_width
        )

        height = (
            int(image_height)
            if image_height is not None
            else self.image_height
        )

        image_points = (
            normalized_points_to_pixels(
                normalized_landmarks,
                width,
                height,
            )
        )

        (
            success,
            rvec,
            tvec,
            inlier_count,
            reprojection_error,
        ) = self._solve_pnp(
            image_points
        )

        hand_scale = estimate_hand_scale(
            normalized_landmarks
        )

        if not success:
            return WristPoseEstimate(
                valid=False,
                rotation_matrix=identity_rotation,
                quaternion_xyzw=np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
                translation=zero_vector,
                rvec=rvec,
                tvec=tvec,
                reprojection_error=reprojection_error,
                inlier_count=inlier_count,
                pnp_weight=0.0,
                scale_depth=0.0,
                hand_scale=hand_scale,
                message="PnP failed.",
            )

        pnp_rotation, _ = cv2.Rodrigues(
            rvec
        )

        pnp_rotation = np.asarray(
            pnp_rotation,
            dtype=np.float64,
        )

        landmark_rotation = (
            estimate_landmark_palm_frame(
                normalized_landmarks
            )
        )

        if self.landmark_alignment is None:
            self.landmark_alignment = (
                pnp_rotation
                @ landmark_rotation.T
            )

        aligned_landmark_rotation = (
            self.landmark_alignment
            @ landmark_rotation
        )

        pnp_weight = float(
            np.clip(
                1.0
                - reprojection_error
                / self.maximum_reprojection_error,
                0.25,
                0.90,
            )
        )

        pnp_quaternion = (
            rotation_matrix_to_quaternion_xyzw(
                pnp_rotation
            )
        )

        landmark_quaternion = (
            rotation_matrix_to_quaternion_xyzw(
                aligned_landmark_rotation
            )
        )

        fused_quaternion = (
            slerp_quaternion_xyzw(
                first=landmark_quaternion,
                second=pnp_quaternion,
                second_weight=pnp_weight,
            )
        )

        fused_rotation = (
            quaternion_xyzw_to_rotation_matrix(
                fused_quaternion
            )
        )

        raw_translation = tvec.reshape(
            3
        ).copy()

        if self.reference_pnp_depth is None:
            self.reference_pnp_depth = float(
                raw_translation[2]
            )

        scale_depth = float(
            self.reference_pnp_depth
            * self.reference_hand_scale
            / max(
                hand_scale,
                1e-8,
            )
        )

        fused_translation = (
            raw_translation.copy()
        )

        fused_translation[2] = (
            pnp_weight
            * raw_translation[2]
            + (1.0 - pnp_weight)
            * scale_depth
        )

        valid = (
            np.isfinite(
                reprojection_error
            )
            and reprojection_error
            <= self.maximum_reprojection_error
            and inlier_count >= 4
            and np.all(
                np.isfinite(
                    fused_translation
                )
            )
            and np.all(
                np.isfinite(
                    fused_rotation
                )
            )
        )

        if valid:
            self.previous_rvec = (
                rvec.copy()
            )

            self.previous_tvec = (
                tvec.copy()
            )

            message = "Pose estimated."

        else:
            message = (
                "Pose rejected because of "
                "reprojection error or invalid values."
            )

        return WristPoseEstimate(
            valid=valid,
            rotation_matrix=fused_rotation,
            quaternion_xyzw=fused_quaternion,
            translation=fused_translation,
            rvec=rvec,
            tvec=tvec,
            reprojection_error=reprojection_error,
            inlier_count=inlier_count,
            pnp_weight=pnp_weight,
            scale_depth=scale_depth,
            hand_scale=hand_scale,
            message=message,
        )