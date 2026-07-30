from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import gymnasium as gym
import mediapipe as mp
import numpy as np
import pybullet as p

import panda_gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.teleop.constrained_ik_controller import (
    ConstrainedIKController,
)
from src.teleop.monocular_wrist_pose import (
    MonocularWristPoseEstimator,
    WristPoseEstimate,
)
from src.teleop.pose_filter import (
    FilteredPose,
    PoseFilter,
)
from src.teleop.pose_mapper import (
    MappedPose,
    PoseMapper,
)


# ============================================================
# 配置
# ============================================================

ENVIRONMENT_ID = "PandaAllegroPickAndPlace-v0"

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

SIMULATION_STEPS_PER_CAMERA_FRAME = 4

# 手指目标滤波。
# 旧版映射本身延迟较低，这里只保留较轻滤波。
HAND_SMOOTHING_ALPHA = 0.45

ROTATION_TRACKING_DEFAULT = True


# ============================================================
# 通用函数
# ============================================================

def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm < 1e-8:
        return np.zeros_like(
            vector
        )

    return vector / norm


def landmark_to_array(
    landmark,
) -> np.ndarray:
    return np.array(
        [
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
        ],
        dtype=np.float64,
    )


def safe_angle_degrees(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
) -> float:
    first_vector = np.asarray(
        first_vector,
        dtype=np.float64,
    )

    second_vector = np.asarray(
        second_vector,
        dtype=np.float64,
    )

    first_norm = float(
        np.linalg.norm(first_vector)
    )

    second_norm = float(
        np.linalg.norm(second_vector)
    )

    if (
        first_norm < 1e-6
        or second_norm < 1e-6
    ):
        return 180.0

    cosine_value = float(
        np.dot(
            first_vector,
            second_vector,
        )
        / (
            first_norm
            * second_norm
        )
    )

    cosine_value = float(
        np.clip(
            cosine_value,
            -1.0,
            1.0,
        )
    )

    return float(
        math.degrees(
            math.acos(
                cosine_value
            )
        )
    )


def draw_text(
    frame: np.ndarray,
    text: str,
    line_index: int,
    color: Tuple[int, int, int],
) -> None:
    y_position = (
        26
        + line_index * 24
    )

    cv2.putText(
        frame,
        text,
        (13, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
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
        results
        .multi_hand_landmarks[0]
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
            results
            .multi_hand_world_landmarks[0]
            .landmark
        )

    return (
        normalized_landmarks,
        world_landmarks,
    )


def step_simulation(
    simulation,
    step_count: int,
) -> None:
    for _ in range(
        step_count
    ):
        if hasattr(
            simulation,
            "step",
        ):
            simulation.step()

        else:
            p.stepSimulation()


# ============================================================
# 旧版 MediaPipe 手指弯曲估计
# ============================================================

def estimate_finger_curl(
    landmarks: Sequence,
    landmark_ids: Sequence[int],
) -> float:
    """
    输入：
        [MCP, PIP, DIP, TIP]

    输出：
        0.0 = 张开
        1.0 = 弯曲
    """
    mcp = landmark_to_array(
        landmarks[
            landmark_ids[0]
        ]
    )

    pip = landmark_to_array(
        landmarks[
            landmark_ids[1]
        ]
    )

    dip = landmark_to_array(
        landmarks[
            landmark_ids[2]
        ]
    )

    tip = landmark_to_array(
        landmarks[
            landmark_ids[3]
        ]
    )

    first_vector = (
        mcp
        - pip
    )

    second_vector = (
        tip
        - pip
    )

    angle = safe_angle_degrees(
        first_vector,
        second_vector,
    )

    curl_from_angle = (
        165.0
        - angle
    ) / (
        165.0
        - 70.0
    )

    curl_from_angle = float(
        np.clip(
            curl_from_angle,
            0.0,
            1.0,
        )
    )

    wrist = landmark_to_array(
        landmarks[0]
    )

    fingertip_distance = float(
        np.linalg.norm(
            tip
            - wrist
        )
    )

    mcp_distance = float(
        np.linalg.norm(
            mcp
            - wrist
        )
    ) + 1e-6

    distance_ratio = (
        fingertip_distance
        / mcp_distance
    )

    curl_from_distance = (
        1.0
        - np.clip(
            (
                distance_ratio
                - 1.1
            )
            / (
                2.2
                - 1.1
            ),
            0.0,
            1.0,
        )
    )

    curl = (
        0.75
        * curl_from_angle
        + 0.25
        * curl_from_distance
    )

    return float(
        np.clip(
            curl,
            0.0,
            1.0,
        )
    )


def estimate_all_finger_curls(
    landmarks: Sequence,
) -> np.ndarray:
    index_curl = estimate_finger_curl(
        landmarks,
        [
            5,
            6,
            7,
            8,
        ],
    )

    middle_curl = estimate_finger_curl(
        landmarks,
        [
            9,
            10,
            11,
            12,
        ],
    )

    ring_curl = estimate_finger_curl(
        landmarks,
        [
            13,
            14,
            15,
            16,
        ],
    )

    pinky_curl = estimate_finger_curl(
        landmarks,
        [
            17,
            18,
            19,
            20,
        ],
    )

    thumb_curl = estimate_finger_curl(
        landmarks,
        [
            1,
            2,
            3,
            4,
        ],
    )

    # Allegro 只有三根普通手指。
    # 第三根使用人手 ring 和 pinky 的组合。
    allegro_ring_curl = (
        0.70
        * ring_curl
        + 0.30
        * pinky_curl
    )

    return np.array(
        [
            index_curl,
            middle_curl,
            allegro_ring_curl,
            thumb_curl,
        ],
        dtype=np.float64,
    )


def curls_to_allegro_target_angles(
    curls: np.ndarray,
    robot,
) -> np.ndarray:
    curls = np.asarray(
        curls,
        dtype=np.float64,
    ).reshape(4)

    lower_limits = np.asarray(
        robot.allegro_lower_limits,
        dtype=np.float64,
    ).reshape(16)

    upper_limits = np.asarray(
        robot.allegro_upper_limits,
        dtype=np.float64,
    ).reshape(16)

    open_angles = np.clip(
        np.zeros(
            16,
            dtype=np.float64,
        ),
        lower_limits,
        upper_limits,
    )

    normal_finger_weights = np.array(
        [
            0.12,
            0.80,
            0.85,
            0.85,
        ],
        dtype=np.float64,
    )

    thumb_weights = np.array(
        [
            0.35,
            0.65,
            0.75,
            0.80,
        ],
        dtype=np.float64,
    )

    target_angles = (
        open_angles.copy()
    )

    for finger_index in range(
        3
    ):
        start_index = (
            finger_index
            * 4
        )

        end_index = (
            start_index
            + 4
        )

        target_angles[
            start_index:end_index
        ] = (
            open_angles[
                start_index:end_index
            ]
            + curls[
                finger_index
            ]
            * normal_finger_weights
            * (
                upper_limits[
                    start_index:end_index
                ]
                - open_angles[
                    start_index:end_index
                ]
            )
        )

    target_angles[
        12:16
    ] = (
        open_angles[
            12:16
        ]
        + curls[3]
        * thumb_weights
        * (
            upper_limits[
                12:16
            ]
            - open_angles[
                12:16
            ]
        )
    )

    return np.clip(
        target_angles,
        lower_limits,
        upper_limits,
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Camera -> PandaAllegro full teleoperation")
    print("=" * 80)
    print("Environment:", ENVIRONMENT_ID)
    print("Intrinsics:", INTRINSICS_PATH)
    print("Mapping:", MAPPING_CONFIG_PATH)
    print("Finger mapping: restored MediaPipe curl mapping")
    print("Finger control: independent from Panda IK")

    if not INTRINSICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing camera intrinsics: "
            f"{INTRINSICS_PATH}"
        )

    if not MAPPING_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing mapping config: "
            f"{MAPPING_CONFIG_PATH}"
        )

    environment = gym.make(
        ENVIRONMENT_ID,
        render_mode="human",
        control_type="ee",
    )

    capture = None

    try:
        environment.reset()

        unwrapped_environment = (
            environment.unwrapped
        )

        robot = (
            unwrapped_environment.robot
        )

        simulation = (
            unwrapped_environment.sim
        )

        for _ in range(
            120
        ):
            step_simulation(
                simulation,
                1,
            )

        ik_controller = (
            ConstrainedIKController(
                robot=robot,
                maximum_joint_step=0.15,
                maximum_position_error=0.025,
                maximum_orientation_error_degrees=12.0,
                soft_limit_margin=0.05,
                joint_damping=0.05,
                maximum_iterations=120,
                residual_threshold=1e-5,
            )
        )

        estimator = (
            MonocularWristPoseEstimator(
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
        )

        mapper = PoseMapper()

        mapper.load_configuration(
            str(
                MAPPING_CONFIG_PATH
            )
        )

        pose_filter = PoseFilter(
            position_min_cutoff=1.0,
            position_beta=0.03,
            position_derivative_cutoff=1.0,
            rotation_min_cutoff=1.5,
            rotation_beta=0.05,
            rotation_derivative_cutoff=1.0,
            maximum_linear_velocity=0.18,
            maximum_linear_acceleration=0.70,
            maximum_angular_velocity=1.10,
            maximum_angular_acceleration=3.50,
            workspace_half_extent=np.array(
                [
                    0.22,
                    0.22,
                    0.20,
                ],
                dtype=np.float64,
            ),
            minimum_z=0.08,
            maximum_relative_rotation_degrees=100.0,
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

        mp_hands = (
            mp.solutions.hands
        )

        drawing_utils = (
            mp.solutions.drawing_utils
        )

        drawing_styles = (
            mp.solutions.drawing_styles
        )

        calibration_active = True
        reference_pending = True

        rotation_tracking = (
            ROTATION_TRACKING_DEFAULT
        )

        teleoperation_paused = False

        filtered_hand_targets = np.asarray(
            ik_controller
            .get_current_allegro_joint_angles(),
            dtype=np.float64,
        )

        last_arm_targets = np.asarray(
            ik_controller
            .get_current_arm_joint_angles(),
            dtype=np.float64,
        )

        last_finger_curls = np.zeros(
            4,
            dtype=np.float64,
        )

        accepted_count = 0
        rejected_count = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        ) as hands:
            while True:
                frame_success, frame = (
                    capture.read()
                )

                if not frame_success:
                    print(
                        "Cannot read webcam frame."
                    )
                    break

                current_timestamp = (
                    time.monotonic()
                )

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

                display_frame = (
                    frame.copy()
                )

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

                filtered_pose: Optional[
                    FilteredPose
                ] = None

                ik_result = None

                # ====================================================
                # Wrist pose calibration and estimation
                # ====================================================

                if normalized_landmarks is not None:
                    if calibration_active:
                        calibration_completed = (
                            estimator.add_calibration_frame(
                                normalized_landmarks=(
                                    normalized_landmarks
                                ),
                                world_landmarks=(
                                    world_landmarks
                                ),
                            )
                        )

                        if calibration_completed:
                            calibration_active = (
                                False
                            )

                            reference_pending = (
                                True
                            )

                            print(
                                "Wrist calibration completed."
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

                # ====================================================
                # Set Camera -> Robot reference
                # ====================================================

                if (
                    estimate is not None
                    and estimate.valid
                    and reference_pending
                ):
                    (
                        robot_reference_position,
                        robot_reference_quaternion,
                    ) = (
                        ik_controller
                        .get_current_ee_pose()
                    )

                    mapper.set_reference(
                        camera_position=(
                            estimate.translation
                        ),
                        camera_rotation=(
                            estimate.rotation_matrix
                        ),
                        robot_position=(
                            robot_reference_position
                        ),
                        robot_rotation=(
                            robot_reference_quaternion
                        ),
                    )

                    mapped_reference = (
                        mapper.map_pose(
                            camera_position=(
                                estimate.translation
                            ),
                            camera_rotation=(
                                estimate.rotation_matrix
                            ),
                        )
                    )

                    pose_filter.set_reference(
                        position=(
                            mapped_reference.position
                        ),
                        rotation=(
                            mapped_reference.rotation_matrix
                        ),
                        timestamp=(
                            current_timestamp
                        ),
                    )

                    if hasattr(
                        ik_controller,
                        "reset_continuity_reference",
                    ):
                        ik_controller.reset_continuity_reference()

                    last_arm_targets = np.asarray(
                        ik_controller
                        .get_current_arm_joint_angles(),
                        dtype=np.float64,
                    )

                    reference_pending = (
                        False
                    )

                    print(
                        "Robot teleoperation reference set."
                    )

                    print(
                        "robot_reference_position:",
                        np.round(
                            robot_reference_position,
                            5,
                        ),
                    )

                    print(
                        "robot_reference_quaternion:",
                        np.round(
                            robot_reference_quaternion,
                            5,
                        ),
                    )

                # ====================================================
                # Wrist mapping and filtering
                # ====================================================

                tracking_valid = (
                    estimate is not None
                    and estimate.valid
                    and mapper.is_calibrated
                    and not reference_pending
                    and not teleoperation_paused
                )

                if tracking_valid:
                    mapped_pose = (
                        mapper.map_pose(
                            camera_position=(
                                estimate.translation
                            ),
                            camera_rotation=(
                                estimate.rotation_matrix
                            ),
                        )
                    )

                    if mapped_pose.valid:
                        if rotation_tracking:
                            target_rotation = (
                                mapped_pose.rotation_matrix
                            )

                        else:
                            target_rotation = (
                                pose_filter.previous_rotation
                            )

                        filtered_pose = (
                            pose_filter.filter_pose(
                                target_position=(
                                    mapped_pose.position
                                ),
                                target_rotation=(
                                    target_rotation
                                ),
                                timestamp=(
                                    current_timestamp
                                ),
                                input_valid=True,
                            )
                        )

                elif pose_filter.is_initialized:
                    filtered_pose = (
                        pose_filter.filter_pose(
                            target_position=(
                                pose_filter.previous_position
                            ),
                            target_rotation=(
                                pose_filter.previous_rotation
                            ),
                            timestamp=(
                                current_timestamp
                            ),
                            input_valid=False,
                        )
                    )

                # ====================================================
                # Allegro finger mapping
                #
                # 独立于 Panda IK。
                # 即使 IK 被拒绝，手指仍然继续更新。
                # ====================================================

                if (
                    normalized_landmarks
                    is not None
                    and not teleoperation_paused
                ):
                    raw_finger_curls = (
                        estimate_all_finger_curls(
                            normalized_landmarks
                        )
                    )

                    raw_hand_targets = (
                        curls_to_allegro_target_angles(
                            raw_finger_curls,
                            robot,
                        )
                    )

                    filtered_hand_targets = (
                        HAND_SMOOTHING_ALPHA
                        * raw_hand_targets
                        + (
                            1.0
                            - HAND_SMOOTHING_ALPHA
                        )
                        * filtered_hand_targets
                    )

                    last_finger_curls = (
                        raw_finger_curls.copy()
                    )

                # ====================================================
                # Panda IK
                # ====================================================

                if (
                    filtered_pose is not None
                    and filtered_pose.valid
                    and not teleoperation_paused
                ):
                    ik_result = (
                        ik_controller.solve(
                            target_position=(
                                filtered_pose.position
                            ),
                            target_quaternion_xyzw=(
                                filtered_pose
                                .quaternion_xyzw
                            ),
                        )
                    )

                    if ik_result.valid:
                        last_arm_targets = (
                            ik_result
                            .arm_joint_targets
                            .copy()
                        )

                        accepted_count += 1

                    else:
                        rejected_count += 1

                # ====================================================
                # Send robot command
                #
                # IK valid:
                #     newest arm + newest hand
                #
                # IK invalid:
                #     previous arm + newest hand
                # ====================================================

                if not teleoperation_paused:
                    all_joint_targets = (
                        np.concatenate(
                            [
                                last_arm_targets,
                                filtered_hand_targets,
                            ]
                        )
                    )

                    robot.control_joints(
                        target_angles=(
                            all_joint_targets
                        )
                    )

                step_simulation(
                    simulation,
                    SIMULATION_STEPS_PER_CAMERA_FRAME,
                )

                # ====================================================
                # Camera overlay
                # ====================================================

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
                            "Keep palm open "
                            "and wrist still"
                        ),
                        1,
                        (0, 255, 255),
                    )

                elif reference_pending:
                    draw_text(
                        display_frame,
                        (
                            "Waiting for valid "
                            "robot reference"
                        ),
                        0,
                        (0, 255, 255),
                    )

                else:
                    if teleoperation_paused:
                        status_text = (
                            "TELEOP: PAUSED"
                        )

                        status_color = (
                            0,
                            255,
                            255,
                        )

                    elif (
                        estimate is not None
                        and estimate.valid
                    ):
                        status_text = (
                            "TELEOP: TRACKING"
                        )

                        status_color = (
                            0,
                            255,
                            0,
                        )

                    else:
                        status_text = (
                            "TELEOP: HOLDING"
                        )

                        status_color = (
                            0,
                            165,
                            255,
                        )

                    draw_text(
                        display_frame,
                        status_text,
                        0,
                        status_color,
                    )

                    draw_text(
                        display_frame,
                        (
                            "rotation tracking: "
                            f"{rotation_tracking}"
                        ),
                        1,
                        (
                            0,
                            255,
                            0,
                        )
                        if rotation_tracking
                        else (
                            0,
                            255,
                            255,
                        ),
                    )

                    if filtered_pose is not None:
                        draw_text(
                            display_frame,
                            (
                                "target XYZ: "
                                f"["
                                f"{filtered_pose.position[0]:.3f}, "
                                f"{filtered_pose.position[1]:.3f}, "
                                f"{filtered_pose.position[2]:.3f}"
                                f"]"
                            ),
                            2,
                            (255, 255, 0),
                        )

                    if estimate is not None:
                        draw_text(
                            display_frame,
                            (
                                "reprojection error: "
                                f"{estimate.reprojection_error:.2f}px"
                            ),
                            3,
                            (
                                0,
                                255,
                                0,
                            )
                            if estimate.valid
                            else (
                                0,
                                0,
                                255,
                            ),
                        )

                    if ik_result is not None:
                        draw_text(
                            display_frame,
                            (
                                "IK: "
                                + (
                                    "ACCEPTED"
                                    if ik_result.valid
                                    else "REJECTED"
                                )
                            ),
                            4,
                            (
                                0,
                                255,
                                0,
                            )
                            if ik_result.valid
                            else (
                                0,
                                0,
                                255,
                            ),
                        )

                        draw_text(
                            display_frame,
                            (
                                "position error: "
                                f"{ik_result.position_error:.4f}"
                            ),
                            5,
                            (255, 255, 255),
                        )

                        draw_text(
                            display_frame,
                            (
                                "orientation error: "
                                f"{ik_result.orientation_error_degrees:.2f} deg"
                            ),
                            6,
                            (255, 255, 255),
                        )

                        draw_text(
                            display_frame,
                            (
                                "maximum joint step: "
                                f"{ik_result.maximum_joint_step:.4f}"
                            ),
                            7,
                            (255, 255, 255),
                        )

                    draw_text(
                        display_frame,
                        (
                            "finger curls I/M/R/T: "
                            f"{last_finger_curls[0]:.2f} "
                            f"{last_finger_curls[1]:.2f} "
                            f"{last_finger_curls[2]:.2f} "
                            f"{last_finger_curls[3]:.2f}"
                        ),
                        8,
                        (255, 255, 0),
                    )

                    draw_text(
                        display_frame,
                        (
                            "accepted/rejected: "
                            f"{accepted_count}/"
                            f"{rejected_count}"
                        ),
                        9,
                        (255, 255, 255),
                    )

                draw_text(
                    display_frame,
                    (
                        "C: recalibrate | "
                        "R: reset reference | "
                        "T: rotation | "
                        "SPACE: pause | Q: quit"
                    ),
                    18,
                    (255, 255, 255),
                )

                cv2.imshow(
                    (
                        "Camera PandaAllegro "
                        "Teleoperation"
                    ),
                    display_frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

                if key in (
                    ord("t"),
                    ord("T"),
                ):
                    rotation_tracking = (
                        not rotation_tracking
                    )

                    print(
                        "rotation_tracking:",
                        rotation_tracking,
                    )

                if key == ord(" "):
                    teleoperation_paused = (
                        not teleoperation_paused
                    )

                    if (
                        not teleoperation_paused
                        and hasattr(
                            ik_controller,
                            "reset_continuity_reference",
                        )
                    ):
                        ik_controller.reset_continuity_reference()

                        last_arm_targets = np.asarray(
                            ik_controller
                            .get_current_arm_joint_angles(),
                            dtype=np.float64,
                        )

                    print(
                        "teleoperation_paused:",
                        teleoperation_paused,
                    )

                if key in (
                    ord("r"),
                    ord("R"),
                ):
                    mapper.clear_reference()
                    pose_filter.reset()

                    if hasattr(
                        ik_controller,
                        "reset_continuity_reference",
                    ):
                        ik_controller.reset_continuity_reference()

                    last_arm_targets = np.asarray(
                        ik_controller
                        .get_current_arm_joint_angles(),
                        dtype=np.float64,
                    )

                    reference_pending = (
                        True
                    )

                    print(
                        "Robot reference reset."
                    )

                if key in (
                    ord("c"),
                    ord("C"),
                ):
                    estimator.reset_calibration()
                    mapper.clear_reference()
                    pose_filter.reset()

                    if hasattr(
                        ik_controller,
                        "reset_continuity_reference",
                    ):
                        ik_controller.reset_continuity_reference()

                    last_arm_targets = np.asarray(
                        ik_controller
                        .get_current_arm_joint_angles(),
                        dtype=np.float64,
                    )

                    calibration_active = (
                        True
                    )

                    reference_pending = (
                        True
                    )

                    print(
                        "Full calibration restarted."
                    )

    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )

    finally:
        if capture is not None:
            capture.release()

        cv2.destroyAllWindows()

        environment.close()


if __name__ == "__main__":
    main()
