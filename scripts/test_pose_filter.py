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
    sys.path.insert(0, str(PROJECT_ROOT))


from src.teleop.monocular_wrist_pose import (
    MonocularWristPoseEstimator,
    WristPoseEstimate,
)
from src.teleop.pose_filter import (
    FilteredPose,
    PoseFilter,
    rotation_matrix_to_rotvec,
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
    y_position = 27 + line_index * 24

    cv2.putText(
        frame,
        text,
        (13, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        color,
        2,
        cv2.LINE_AA,
    )


def get_first_hand(results):
    if not results.multi_hand_landmarks:
        return None, None

    normalized_landmarks = (
        results.multi_hand_landmarks[0].landmark
    )

    world_landmarks = None

    if (
        results.multi_hand_world_landmarks
        and len(results.multi_hand_world_landmarks) > 0
    ):
        world_landmarks = (
            results.multi_hand_world_landmarks[0].landmark
        )

    return normalized_landmarks, world_landmarks


def rotation_vector_degrees(
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    return (
        rotation_matrix_to_rotvec(
            rotation_matrix
        )
        * 180.0
        / np.pi
    )


def draw_position_indicator(
    frame: np.ndarray,
    raw_position: np.ndarray,
    filtered_position: np.ndarray,
) -> None:
    height, width = frame.shape[:2]

    panel_center = np.array(
        [width - 115, height - 110],
        dtype=np.int32,
    )

    position_scale = 420.0

    raw_delta = (
        raw_position
        - VIRTUAL_ROBOT_REFERENCE_POSITION
    )

    filtered_delta = (
        filtered_position
        - VIRTUAL_ROBOT_REFERENCE_POSITION
    )

    raw_point = panel_center + np.array(
        [
            int(
                np.clip(
                    -raw_delta[1] * position_scale,
                    -85,
                    85,
                )
            ),
            int(
                np.clip(
                    -raw_delta[2] * position_scale,
                    -85,
                    85,
                )
            ),
        ],
        dtype=np.int32,
    )

    filtered_point = panel_center + np.array(
        [
            int(
                np.clip(
                    -filtered_delta[1] * position_scale,
                    -85,
                    85,
                )
            ),
            int(
                np.clip(
                    -filtered_delta[2] * position_scale,
                    -85,
                    85,
                )
            ),
        ],
        dtype=np.int32,
    )

    cv2.circle(
        frame,
        tuple(panel_center),
        4,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        tuple(raw_point),
        7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        tuple(filtered_point),
        6,
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        tuple(panel_center),
        tuple(filtered_point),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "red: raw",
        (width - 210, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "green: filtered",
        (width - 210, height - 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    print("=" * 80)
    print("Pose filter test")
    print("=" * 80)
    print("Intrinsics:", INTRINSICS_PATH)
    print("Mapping:", MAPPING_CONFIG_PATH)

    if not MAPPING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Mapping configuration does not exist: "
            f"{MAPPING_CONFIG_PATH}"
        )

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

    mapper.load_configuration(
        str(MAPPING_CONFIG_PATH)
    )

    pose_filter = PoseFilter(
        position_min_cutoff=1.0,
        position_beta=0.03,
        position_derivative_cutoff=1.0,
        rotation_min_cutoff=1.5,
        rotation_beta=0.05,
        rotation_derivative_cutoff=1.0,
        maximum_linear_velocity=0.20,
        maximum_linear_acceleration=0.80,
        maximum_angular_velocity=1.20,
        maximum_angular_acceleration=4.00,
        workspace_half_extent=np.array(
            [0.25, 0.25, 0.25],
            dtype=np.float64,
        ),
        minimum_z=0.20,
        maximum_relative_rotation_degrees=120.0,
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
    reference_pending = True

    last_console_time = 0.0

    last_raw_position: Optional[
        np.ndarray
    ] = None

    last_raw_rotation: Optional[
        np.ndarray
    ] = None

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

                current_timestamp = time.monotonic()

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
                        results.multi_hand_landmarks[0],
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

                filtered_pose: Optional[
                    FilteredPose
                ] = None

                if normalized_landmarks is not None:
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

                    mapped_reference = mapper.map_pose(
                        camera_position=(
                            estimate.translation
                        ),
                        camera_rotation=(
                            estimate.rotation_matrix
                        ),
                    )

                    pose_filter.set_reference(
                        position=(
                            mapped_reference.position
                        ),
                        rotation=(
                            mapped_reference.rotation_matrix
                        ),
                        timestamp=current_timestamp,
                    )

                    last_raw_position = (
                        mapped_reference.position.copy()
                    )

                    last_raw_rotation = (
                        mapped_reference.rotation_matrix.copy()
                    )

                    reference_pending = False

                    print(
                        "Pose filter reference set."
                    )

                    print(
                        "reference_position:",
                        np.round(
                            mapped_reference.position,
                            5,
                        ),
                    )

                if (
                    estimate is not None
                    and estimate.valid
                    and mapper.is_calibrated
                    and not reference_pending
                ):
                    mapped_pose = mapper.map_pose(
                        camera_position=(
                            estimate.translation
                        ),
                        camera_rotation=(
                            estimate.rotation_matrix
                        ),
                    )

                    if mapped_pose.valid:
                        last_raw_position = (
                            mapped_pose.position.copy()
                        )

                        last_raw_rotation = (
                            mapped_pose.rotation_matrix.copy()
                        )

                        filtered_pose = (
                            pose_filter.filter_pose(
                                target_position=(
                                    mapped_pose.position
                                ),
                                target_rotation=(
                                    mapped_pose.rotation_matrix
                                ),
                                timestamp=(
                                    current_timestamp
                                ),
                                input_valid=True,
                            )
                        )

                elif (
                    pose_filter.is_initialized
                    and last_raw_position is not None
                    and last_raw_rotation is not None
                ):
                    filtered_pose = (
                        pose_filter.filter_pose(
                            target_position=(
                                last_raw_position
                            ),
                            target_rotation=(
                                last_raw_rotation
                            ),
                            timestamp=(
                                current_timestamp
                            ),
                            input_valid=False,
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

                elif reference_pending:
                    draw_text(
                        display_frame,
                        "Waiting for valid reference pose",
                        0,
                        (0, 255, 255),
                    )

                elif filtered_pose is None:
                    draw_text(
                        display_frame,
                        "No filtered pose",
                        0,
                        (0, 0, 255),
                    )

                else:
                    status_color = (
                        (0, 255, 0)
                        if filtered_pose.valid
                        else (0, 255, 255)
                    )

                    draw_text(
                        display_frame,
                        (
                            "FILTER: "
                            + (
                                "TRACKING"
                                if filtered_pose.valid
                                else "HOLDING"
                            )
                        ),
                        0,
                        status_color,
                    )

                    raw_position = (
                        last_raw_position
                        if last_raw_position is not None
                        else filtered_pose.position
                    )

                    raw_rotation = (
                        last_raw_rotation
                        if last_raw_rotation is not None
                        else filtered_pose.rotation_matrix
                    )

                    raw_rotation_degrees = (
                        rotation_vector_degrees(
                            raw_rotation
                        )
                    )

                    filtered_rotation_degrees = (
                        rotation_vector_degrees(
                            filtered_pose.rotation_matrix
                        )
                    )

                    draw_text(
                        display_frame,
                        (
                            "raw XYZ: "
                            f"[{raw_position[0]:.3f}, "
                            f"{raw_position[1]:.3f}, "
                            f"{raw_position[2]:.3f}]"
                        ),
                        1,
                        (0, 0, 255),
                    )

                    draw_text(
                        display_frame,
                        (
                            "filtered XYZ: "
                            f"[{filtered_pose.position[0]:.3f}, "
                            f"{filtered_pose.position[1]:.3f}, "
                            f"{filtered_pose.position[2]:.3f}]"
                        ),
                        2,
                        (0, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "linear velocity: "
                            f"{np.linalg.norm(filtered_pose.linear_velocity):.3f} m/s"
                        ),
                        3,
                        (255, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "angular velocity: "
                            f"{np.linalg.norm(filtered_pose.angular_velocity):.3f} rad/s"
                        ),
                        4,
                        (255, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "raw rotvec deg: "
                            f"[{raw_rotation_degrees[0]:+.1f}, "
                            f"{raw_rotation_degrees[1]:+.1f}, "
                            f"{raw_rotation_degrees[2]:+.1f}]"
                        ),
                        5,
                        (0, 0, 255),
                    )

                    draw_text(
                        display_frame,
                        (
                            "filtered rotvec deg: "
                            f"[{filtered_rotation_degrees[0]:+.1f}, "
                            f"{filtered_rotation_degrees[1]:+.1f}, "
                            f"{filtered_rotation_degrees[2]:+.1f}]"
                        ),
                        6,
                        (0, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "position clipped: "
                            f"{filtered_pose.position_clipped}"
                        ),
                        7,
                        (
                            (0, 165, 255)
                            if filtered_pose.position_clipped
                            else (255, 255, 255)
                        ),
                    )

                    draw_text(
                        display_frame,
                        (
                            "rotation clipped: "
                            f"{filtered_pose.rotation_clipped}"
                        ),
                        8,
                        (
                            (0, 165, 255)
                            if filtered_pose.rotation_clipped
                            else (255, 255, 255)
                        ),
                    )

                    draw_position_indicator(
                        display_frame,
                        raw_position,
                        filtered_pose.position,
                    )

                    if (
                        current_timestamp
                        - last_console_time
                        >= 1.0
                    ):
                        print(
                            "valid=",
                            filtered_pose.valid,
                            "raw=",
                            np.round(
                                raw_position,
                                4,
                            ),
                            "filtered=",
                            np.round(
                                filtered_pose.position,
                                4,
                            ),
                            "linear_speed=",
                            round(
                                float(
                                    np.linalg.norm(
                                        filtered_pose.linear_velocity
                                    )
                                ),
                                4,
                            ),
                            "angular_speed=",
                            round(
                                float(
                                    np.linalg.norm(
                                        filtered_pose.angular_velocity
                                    )
                                ),
                                4,
                            ),
                            "pos_clip=",
                            filtered_pose.position_clipped,
                            "rot_clip=",
                            filtered_pose.rotation_clipped,
                        )

                        last_console_time = (
                            current_timestamp
                        )

                draw_text(
                    display_frame,
                    "C: recalibrate | Q: quit",
                    18,
                    (255, 255, 255),
                )

                cv2.imshow(
                    "Pose Filter Test",
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

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
                    pose_filter.reset()

                    calibration_active = True
                    reference_pending = True

                    last_raw_position = None
                    last_raw_rotation = None

                    print(
                        "Calibration restarted."
                    )

        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()