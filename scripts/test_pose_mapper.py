from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

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
from src.teleop.pose_mapper import (
    MappedPose,
    PoseMapper,
)


INTRINSICS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "webcam_intrinsics.npz"
)

MAPPING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "teleop_mapping.npz"
)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CALIBRATION_FRAME_COUNT = 45
MAXIMUM_REPROJECTION_ERROR = 10.0

VIRTUAL_ROBOT_REFERENCE_POSITION = np.array(
    [0.45, 0.00, 0.55],
    dtype=np.float64,
)

VIRTUAL_ROBOT_REFERENCE_ROTATION = np.eye(
    3,
    dtype=np.float64,
)


def draw_text(
    frame: np.ndarray,
    text: str,
    line_index: int,
    color: Tuple[int, int, int],
) -> None:
    y_position = 28 + line_index * 25

    cv2.putText(
        frame,
        text,
        (14, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
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


def rotation_matrix_to_rotvec_degrees(
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(
        np.asarray(
            rotation_matrix,
            dtype=np.float64,
        )
    )

    return (
        rotation_vector.reshape(3)
        * 180.0
        / np.pi
    )


def draw_direction_indicator(
    frame: np.ndarray,
    robot_delta: np.ndarray,
) -> None:
    height, width = frame.shape[:2]

    center = np.array(
        [
            width - 105,
            height - 105,
        ],
        dtype=np.int32,
    )

    scale = 800.0

    horizontal_end = center + np.array(
        [
            int(
                np.clip(
                    -robot_delta[1] * scale,
                    -75,
                    75,
                )
            ),
            int(
                np.clip(
                    -robot_delta[2] * scale,
                    -75,
                    75,
                )
            ),
        ],
        dtype=np.int32,
    )

    cv2.circle(
        frame,
        tuple(center),
        5,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        frame,
        tuple(center),
        tuple(horizontal_end),
        (0, 255, 255),
        3,
        cv2.LINE_AA,
        tipLength=0.20,
    )

    depth_text = (
        f"robot X depth: "
        f"{robot_delta[0]:+.3f}"
    )

    cv2.putText(
        frame,
        depth_text,
        (
            width - 215,
            height - 25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    print("=" * 80)
    print("Pose mapper test")
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

    mapper = PoseMapper()

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
    reference_pending = True
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
                success, frame = capture.read()

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

                rgb_frame.flags.writeable = False

                results = hands.process(
                    rgb_frame
                )

                rgb_frame.flags.writeable = True

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

                mapped_pose: Optional[
                    MappedPose
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
                            calibration_active = False
                            reference_pending = True

                            print(
                                "Wrist calibration completed."
                            )

                    if estimator.is_calibrated:
                        estimate = estimator.estimate(
                            normalized_landmarks=(
                                normalized_landmarks
                            ),
                            world_landmarks=(
                                world_landmarks
                            ),
                            image_width=frame_width,
                            image_height=frame_height,
                        )

                if (
                    estimate is not None
                    and estimate.valid
                    and reference_pending
                ):
                    mapper.set_reference(
                        camera_position=(
                            estimate.translation
                        ),
                        camera_rotation=(
                            estimate.rotation_matrix
                        ),
                        robot_position=(
                            VIRTUAL_ROBOT_REFERENCE_POSITION
                        ),
                        robot_rotation=(
                            VIRTUAL_ROBOT_REFERENCE_ROTATION
                        ),
                    )

                    reference_pending = False

                    print(
                        "Pose mapping reference set."
                    )

                    print(
                        "camera_reference_position:",
                        np.round(
                            estimate.translation,
                            5,
                        ),
                    )

                    print(
                        "robot_reference_position:",
                        VIRTUAL_ROBOT_REFERENCE_POSITION,
                    )

                if (
                    estimate is not None
                    and estimate.valid
                    and mapper.is_calibrated
                ):
                    mapped_pose = mapper.map_pose(
                        camera_position=(
                            estimate.translation
                        ),
                        camera_rotation=(
                            estimate.rotation_matrix
                        ),
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

                elif reference_pending:
                    draw_text(
                        display_frame,
                        "Waiting for valid reference pose",
                        0,
                        (0, 255, 255),
                    )

                elif mapped_pose is None:
                    draw_text(
                        display_frame,
                        "No valid mapped pose",
                        0,
                        (0, 0, 255),
                    )

                else:
                    status_color = (
                        (0, 255, 0)
                        if mapped_pose.valid
                        else (0, 0, 255)
                    )

                    draw_text(
                        display_frame,
                        (
                            "MAPPING: "
                            + (
                                "VALID"
                                if mapped_pose.valid
                                else "REJECTED"
                            )
                        ),
                        0,
                        status_color,
                    )

                    camera_delta = (
                        mapped_pose
                        .camera_delta_position
                    )

                    robot_delta = (
                        mapped_pose
                        .robot_delta_position
                    )

                    target_position = (
                        mapped_pose.position
                    )

                    robot_rotation_degrees = (
                        rotation_matrix_to_rotvec_degrees(
                            mapped_pose
                            .robot_delta_rotation
                        )
                    )

                    draw_text(
                        display_frame,
                        (
                            "camera delta: "
                            f"[{camera_delta[0]:+.3f}, "
                            f"{camera_delta[1]:+.3f}, "
                            f"{camera_delta[2]:+.3f}]"
                        ),
                        1,
                        (255, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "robot delta XYZ: "
                            f"[{robot_delta[0]:+.3f}, "
                            f"{robot_delta[1]:+.3f}, "
                            f"{robot_delta[2]:+.3f}]"
                        ),
                        2,
                        (255, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "target XYZ: "
                            f"[{target_position[0]:.3f}, "
                            f"{target_position[1]:.3f}, "
                            f"{target_position[2]:.3f}]"
                        ),
                        3,
                        (0, 255, 255),
                    )

                    draw_text(
                        display_frame,
                        (
                            "rotation vector deg: "
                            f"[{robot_rotation_degrees[0]:+.1f}, "
                            f"{robot_rotation_degrees[1]:+.1f}, "
                            f"{robot_rotation_degrees[2]:+.1f}]"
                        ),
                        4,
                        (0, 255, 255),
                    )

                    draw_text(
                        display_frame,
                        (
                            "reprojection error: "
                            f"{estimate.reprojection_error:.2f}px"
                        ),
                        5,
                        status_color,
                    )

                    draw_direction_indicator(
                        display_frame,
                        robot_delta,
                    )

                    current_time = time.time()

                    if (
                        current_time
                        - last_console_time
                        >= 1.0
                    ):
                        print(
                            "camera_delta=",
                            np.round(
                                camera_delta,
                                4,
                            ),
                            "robot_delta=",
                            np.round(
                                robot_delta,
                                4,
                            ),
                            "rotation_deg=",
                            np.round(
                                robot_rotation_degrees,
                                2,
                            ),
                        )

                        last_console_time = (
                            current_time
                        )

                draw_text(
                    display_frame,
                    (
                        "C: recalibrate | "
                        "S: save mapping | Q: quit"
                    ),
                    17,
                    (255, 255, 255),
                )

                cv2.imshow(
                    "Pose Mapper Test",
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
                    mapper.clear_reference()

                    calibration_active = True
                    reference_pending = True

                    print(
                        "Calibration restarted."
                    )

                if key in (
                    ord("s"),
                    ord("S"),
                ):
                    mapper.save_configuration(
                        str(
                            MAPPING_CONFIG_PATH
                        )
                    )

                    print(
                        "Mapping saved:",
                        MAPPING_CONFIG_PATH,
                    )

        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()