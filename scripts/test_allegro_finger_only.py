from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import mediapipe as mp
import numpy as np
import pybullet as p

import panda_gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from run_camera_panda_allegro_teleop import (
    estimate_allegro_joint_targets,
    get_first_hand,
    step_simulation,
)


ENVIRONMENT_ID = "PandaAllegroPickAndPlace-v0"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SIMULATION_STEPS_PER_FRAME = 4

HAND_SMOOTHING_ALPHA = 0.45


def draw_text(
    frame: np.ndarray,
    text: str,
    line_index: int,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        text,
        (15, 28 + line_index * 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def get_arm_joint_angles(
    robot,
) -> np.ndarray:
    return np.asarray(
        [
            robot.get_joint_angle(
                joint=int(joint_index)
            )
            for joint_index
            in robot.arm_joint_indices
        ],
        dtype=np.float64,
    )


def summarize_fingers(
    joint_angles: np.ndarray,
) -> tuple[float, float, float, float]:
    joint_angles = np.asarray(
        joint_angles,
        dtype=np.float64,
    ).reshape(16)

    index_value = float(
        np.mean(
            joint_angles[1:4]
        )
    )

    middle_value = float(
        np.mean(
            joint_angles[5:8]
        )
    )

    ring_value = float(
        np.mean(
            joint_angles[9:12]
        )
    )

    thumb_value = float(
        np.mean(
            joint_angles[12:16]
        )
    )

    return (
        index_value,
        middle_value,
        ring_value,
        thumb_value,
    )


def main() -> None:
    print("=" * 80)
    print("Allegro finger-only retargeting test")
    print("=" * 80)

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

        for _ in range(120):
            step_simulation(
                simulation,
                1,
            )

        fixed_arm_targets = (
            get_arm_joint_angles(
                robot
            )
        )

        filtered_hand_targets = np.asarray(
            robot.get_allegro_joint_positions(),
            dtype=np.float64,
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

        last_console_time = 0.0

        frame_count = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        ) as hands:
            while True:
                success, frame = (
                    capture.read()
                )

                if not success:
                    print(
                        "Cannot read webcam frame."
                    )
                    break

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
                    _,
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

                hand_detected = (
                    normalized_landmarks
                    is not None
                )

                if hand_detected:
                    raw_hand_targets = (
                        estimate_allegro_joint_targets(
                            normalized_landmarks,
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

                all_joint_targets = (
                    np.concatenate(
                        [
                            fixed_arm_targets,
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
                    SIMULATION_STEPS_PER_FRAME,
                )

                actual_hand_angles = np.asarray(
                    robot.get_allegro_joint_positions(),
                    dtype=np.float64,
                )

                tracking_error = float(
                    np.max(
                        np.abs(
                            filtered_hand_targets
                            - actual_hand_angles
                        )
                    )
                )

                (
                    target_index,
                    target_middle,
                    target_ring,
                    target_thumb,
                ) = summarize_fingers(
                    filtered_hand_targets
                )

                (
                    actual_index,
                    actual_middle,
                    actual_ring,
                    actual_thumb,
                ) = summarize_fingers(
                    actual_hand_angles
                )

                draw_text(
                    display_frame,
                    (
                        "HAND: TRACKING"
                        if hand_detected
                        else "HAND: HOLDING"
                    ),
                    0,
                    (
                        0,
                        255,
                        0,
                    )
                    if hand_detected
                    else (
                        0,
                        165,
                        255,
                    ),
                )

                draw_text(
                    display_frame,
                    (
                        "target I/M/R/T: "
                        f"{target_index:.2f} "
                        f"{target_middle:.2f} "
                        f"{target_ring:.2f} "
                        f"{target_thumb:.2f}"
                    ),
                    1,
                    (255, 255, 0),
                )

                draw_text(
                    display_frame,
                    (
                        "actual I/M/R/T: "
                        f"{actual_index:.2f} "
                        f"{actual_middle:.2f} "
                        f"{actual_ring:.2f} "
                        f"{actual_thumb:.2f}"
                    ),
                    2,
                    (255, 255, 255),
                )

                draw_text(
                    display_frame,
                    (
                        "maximum tracking error: "
                        f"{tracking_error:.3f}"
                    ),
                    3,
                    (
                        0,
                        255,
                        0,
                    )
                    if tracking_error < 0.20
                    else (
                        0,
                        0,
                        255,
                    ),
                )

                draw_text(
                    display_frame,
                    "Arm fixed | Q: quit",
                    17,
                    (255, 255, 255),
                )

                current_time = (
                    time.monotonic()
                )

                if (
                    current_time
                    - last_console_time
                    >= 1.0
                ):
                    print(
                        "target:",
                        np.round(
                            filtered_hand_targets,
                            2,
                        ),
                    )

                    print(
                        "actual:",
                        np.round(
                            actual_hand_angles,
                            2,
                        ),
                    )

                    print(
                        "maximum error:",
                        round(
                            tracking_error,
                            3,
                        ),
                    )

                    print("-" * 80)

                    last_console_time = (
                        current_time
                    )

                cv2.imshow(
                    "Allegro Finger-Only Test",
                    display_frame,
                )

                frame_count += 1

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

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