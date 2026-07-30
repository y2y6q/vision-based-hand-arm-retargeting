from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.teleop.monocular_wrist_pose import (
    MonocularWristPoseEstimator,
    WristPoseEstimate,
)


INTRINSICS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "webcam_intrinsics.npz"
)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CALIBRATION_FRAME_COUNT = 45
MAXIMUM_REPROJECTION_ERROR = 10.0

AXIS_LENGTH = 0.35


def draw_text(
    frame: np.ndarray,
    text: str,
    line_index: int,
    color: Tuple[int, int, int],
) -> None:
    y = 28 + line_index * 26

    cv2.putText(
        frame,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_pnp_axes(
    frame: np.ndarray,
    estimate: WristPoseEstimate,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    if not estimate.valid:
        return

    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [AXIS_LENGTH, 0.0, 0.0],
            [0.0, AXIS_LENGTH, 0.0],
            [0.0, 0.0, AXIS_LENGTH],
        ],
        dtype=np.float64,
    )

    projected_points, _ = cv2.projectPoints(
        axis_points,
        estimate.rvec,
        estimate.tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected_points = projected_points.reshape(
        -1,
        2,
    )

    if not np.all(
        np.isfinite(
            projected_points
        )
    ):
        return

    points = [
        tuple(
            np.round(point).astype(
                np.int32
            )
        )
        for point in projected_points
    ]

    origin = points[0]

    cv2.line(
        frame,
        origin,
        points[1],
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        origin,
        points[2],
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        origin,
        points[3],
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "X",
        points[1],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Y",
        points[2],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Z",
        points[3],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


def get_first_hand(
    results,
):
    if not results.multi_hand_landmarks:
        return None, None

    normalized_landmarks = (
        results.multi_hand_landmarks[0]
        .landmark
    )

    world_landmarks = None

    if (
        results.multi_hand_world_landmarks
        and len(
            results.multi_hand_world_landmarks
        ) > 0
    ):
        world_landmarks = (
            results.multi_hand_world_landmarks[0]
            .landmark
        )

    return (
        normalized_landmarks,
        world_landmarks,
    )


def main() -> None:
    print("=" * 80)
    print("Monocular wrist pose test")
    print("=" * 80)
    print("Intrinsics:", INTRINSICS_PATH)

    estimator = MonocularWristPoseEstimator(
        intrinsics_path=str(
            INTRINSICS_PATH
        ),
        calibration_frame_count=(
            CALIBRATION_FRAME_COUNT
        ),
        maximum_reprojection_error=(
            MAXIMUM_REPROJECTION_ERROR
        ),
    )

    capture = cv2.VideoCapture(
        CAMERA_INDEX
    )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Cannot open webcam."
        )

    mp_hands = mp.solutions.hands
    drawing_utils = (
        mp.solutions.drawing_utils
    )
    drawing_styles = (
        mp.solutions.drawing_styles
    )

    calibration_active = True
    last_console_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    ) as hands:
        try:
            while True:
                success, frame = (
                    capture.read()
                )

                if not success:
                    print(
                        "Cannot read camera frame."
                    )
                    break

                frame_height, frame_width = (
                    frame.shape[:2]
                )

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                rgb_frame.flags.writeable = (
                    False
                )

                results = hands.process(
                    rgb_frame
                )

                rgb_frame.flags.writeable = (
                    True
                )

                (
                    normalized_landmarks,
                    world_landmarks,
                ) = get_first_hand(
                    results
                )

                display_frame = frame.copy()

                if results.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(
                        display_frame,
                        results.multi_hand_landmarks[
                            0
                        ],
                        mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                estimate: Optional[
                    WristPoseEstimate
                ] = None

                if (
                    normalized_landmarks
                    is not None
                ):
                    if calibration_active:
                        completed = (
                            estimator.add_calibration_frame(
                                normalized_landmarks=(
                                    normalized_landmarks
                                ),
                                world_landmarks=(
                                    world_landmarks
                                ),
                            )
                        )

                        if completed:
                            calibration_active = (
                                False
                            )

                            print(
                                "Calibration completed."
                            )

                    if estimator.is_calibrated:
                        estimate = (
                            estimator.estimate(
                                normalized_landmarks=(
                                    normalized_landmarks
                                ),
                                world_landmarks=(
                                    world_landmarks
                                ),
                                image_width=(
                                    frame_width
                                ),
                                image_height=(
                                    frame_height
                                ),
                            )
                        )

                if calibration_active:
                    current_count, total_count = (
                        estimator.calibration_progress
                    )

                    draw_text(
                        display_frame,
                        (
                            "CALIBRATING: "
                            f"{current_count}/"
                            f"{total_count}"
                        ),
                        0,
                        (0, 255, 255),
                    )

                    draw_text(
                        display_frame,
                        (
                            "Hold palm open "
                            "and keep wrist still"
                        ),
                        1,
                        (0, 255, 255),
                    )

                elif estimate is None:
                    draw_text(
                        display_frame,
                        "No hand detected",
                        0,
                        (0, 0, 255),
                    )

                else:
                    status_color = (
                        (0, 255, 0)
                        if estimate.valid
                        else (0, 0, 255)
                    )

                    draw_text(
                        display_frame,
                        (
                            "POSE: "
                            + (
                                "VALID"
                                if estimate.valid
                                else "REJECTED"
                            )
                        ),
                        0,
                        status_color,
                    )

                    draw_text(
                        display_frame,
                        (
                            "reprojection error: "
                            f"{estimate.reprojection_error:.2f}px"
                        ),
                        1,
                        status_color,
                    )

                    draw_text(
                        display_frame,
                        (
                            "inliers: "
                            f"{estimate.inlier_count}"
                            "  pnp weight: "
                            f"{estimate.pnp_weight:.2f}"
                        ),
                        2,
                        status_color,
                    )

                    translation = (
                        estimate.translation
                    )

                    draw_text(
                        display_frame,
                        (
                            "translation: "
                            f"[{translation[0]:.3f}, "
                            f"{translation[1]:.3f}, "
                            f"{translation[2]:.3f}]"
                        ),
                        3,
                        (255, 255, 0),
                    )

                    quaternion = (
                        estimate.quaternion_xyzw
                    )

                    draw_text(
                        display_frame,
                        (
                            "quat xyzw: "
                            f"[{quaternion[0]:.2f}, "
                            f"{quaternion[1]:.2f}, "
                            f"{quaternion[2]:.2f}, "
                            f"{quaternion[3]:.2f}]"
                        ),
                        4,
                        (255, 255, 0),
                    )

                    draw_pnp_axes(
                        display_frame,
                        estimate,
                        estimator.camera_matrix,
                        estimator.dist_coeffs,
                    )

                    current_time = time.time()

                    if (
                        current_time
                        - last_console_time
                        >= 1.0
                    ):
                        print(
                            "valid=",
                            estimate.valid,
                            "error=",
                            round(
                                estimate.reprojection_error,
                                3,
                            ),
                            "inliers=",
                            estimate.inlier_count,
                            "translation=",
                            np.round(
                                estimate.translation,
                                4,
                            ),
                        )

                        last_console_time = (
                            current_time
                        )

                draw_text(
                    display_frame,
                    "C: recalibrate | Q: quit",
                    16,
                    (255, 255, 255),
                )

                cv2.imshow(
                    "Monocular Wrist Pose Test",
                    display_frame,
                )

                key = cv2.waitKey(
                    1
                ) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

                if key in (
                    ord("c"),
                    ord("C"),
                ):
                    estimator.reset_calibration()
                    calibration_active = True

                    print(
                        "Calibration restarted."
                    )

        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()