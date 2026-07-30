from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import pybullet as p

import panda_gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.teleop.constrained_ik_controller import (
    ConstrainedIKController,
    IKResult,
)


ENVIRONMENT_ID = "PandaAllegroPickAndPlace-v0"

SETTLE_STEPS = 120
STEPS_PER_SEGMENT = 75
SIMULATION_FREQUENCY = 240.0

TRANSLATION_DISTANCE = 0.025
ROTATION_ANGLE_DEGREES = 8.0


def normalize_quaternion(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = np.asarray(
        quaternion,
        dtype=np.float64,
    ).reshape(4)

    norm = float(
        np.linalg.norm(quaternion)
    )

    if norm < 1e-10:
        return np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    return quaternion / norm


def quaternion_multiply(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    first = normalize_quaternion(first)
    second = normalize_quaternion(second)

    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second

    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )

    return normalize_quaternion(result)


def axis_angle_quaternion(
    axis: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    axis = np.asarray(
        axis,
        dtype=np.float64,
    ).reshape(3)

    axis_norm = float(
        np.linalg.norm(axis)
    )

    if axis_norm < 1e-10:
        return np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    axis = axis / axis_norm

    half_angle = math.radians(
        angle_degrees
    ) / 2.0

    sine_value = math.sin(
        half_angle
    )

    return normalize_quaternion(
        np.array(
            [
                axis[0] * sine_value,
                axis[1] * sine_value,
                axis[2] * sine_value,
                math.cos(half_angle),
            ],
            dtype=np.float64,
        )
    )


def quaternion_slerp(
    first: np.ndarray,
    second: np.ndarray,
    interpolation: float,
) -> np.ndarray:
    first = normalize_quaternion(first)
    second = normalize_quaternion(second)

    interpolation = float(
        np.clip(
            interpolation,
            0.0,
            1.0,
        )
    )

    dot_product = float(
        np.dot(
            first,
            second,
        )
    )

    if dot_product < 0.0:
        second = -second
        dot_product = -dot_product

    dot_product = float(
        np.clip(
            dot_product,
            -1.0,
            1.0,
        )
    )

    if dot_product > 0.9995:
        result = (
            first
            + interpolation
            * (second - first)
        )

        return normalize_quaternion(
            result
        )

    initial_angle = math.acos(
        dot_product
    )

    sine_initial_angle = math.sin(
        initial_angle
    )

    first_weight = math.sin(
        (1.0 - interpolation)
        * initial_angle
    ) / sine_initial_angle

    second_weight = math.sin(
        interpolation
        * initial_angle
    ) / sine_initial_angle

    return normalize_quaternion(
        first_weight * first
        + second_weight * second
    )


def step_simulation(
    simulation,
) -> None:
    if hasattr(
        simulation,
        "step",
    ):
        simulation.step()
    else:
        p.stepSimulation()

    time.sleep(
        1.0 / SIMULATION_FREQUENCY
    )


def create_target_marker(
    position: np.ndarray,
) -> int:
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.012,
        rgbaColor=[
            1.0,
            0.0,
            1.0,
            0.85,
        ],
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=position.tolist(),
    )


def update_target_marker(
    marker_id: int,
    position: np.ndarray,
) -> None:
    p.resetBasePositionAndOrientation(
        marker_id,
        position.tolist(),
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )


def run_segment(
    controller: ConstrainedIKController,
    simulation,
    marker_id: int,
    segment_name: str,
    start_position: np.ndarray,
    start_quaternion: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
) -> Tuple[int, int]:
    print()
    print("=" * 80)
    print("Segment:", segment_name)
    print("Target position:", np.round(target_position, 5))
    print("Target quaternion:", np.round(target_quaternion, 5))
    print("=" * 80)

    accepted_count = 0
    rejected_count = 0

    for step_index in range(
        1,
        STEPS_PER_SEGMENT + 1,
    ):
        interpolation = (
            step_index
            / STEPS_PER_SEGMENT
        )

        current_target_position = (
            start_position
            + interpolation
            * (
                target_position
                - start_position
            )
        )

        current_target_quaternion = (
            quaternion_slerp(
                start_quaternion,
                target_quaternion,
                interpolation,
            )
        )

        update_target_marker(
            marker_id,
            current_target_position,
        )

        result: IKResult = (
            controller.solve_and_apply(
                target_position=(
                    current_target_position
                ),
                target_quaternion_xyzw=(
                    current_target_quaternion
                ),
            )
        )

        if result.valid:
            accepted_count += 1
        else:
            rejected_count += 1

        if (
            step_index == 1
            or step_index % 15 == 0
            or step_index
            == STEPS_PER_SEGMENT
        ):
            print(
                f"step={step_index:03d}",
                f"valid={result.valid}",
                f"position_error={result.position_error:.5f}",
                (
                    "orientation_error="
                    f"{result.orientation_error_degrees:.3f}deg"
                ),
                (
                    "maximum_joint_step="
                    f"{result.maximum_joint_step:.5f}"
                ),
                f"message={result.message}",
            )

        step_simulation(
            simulation
        )

    print(
        f"Segment result: accepted={accepted_count}, "
        f"rejected={rejected_count}"
    )

    return accepted_count, rejected_count


def main() -> None:
    print("=" * 80)
    print("Constrained IK controller test")
    print("=" * 80)
    print("Environment:", ENVIRONMENT_ID)

    environment = gym.make(
        ENVIRONMENT_ID,
        render_mode="human",
        control_type="ee",
    )

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
            SETTLE_STEPS
        ):
            step_simulation(
                simulation
            )

        controller = (
            ConstrainedIKController(
                robot=robot,
                maximum_joint_step=0.15,
                maximum_position_error=0.02,
                maximum_orientation_error_degrees=10.0,
                soft_limit_margin=0.05,
                joint_damping=0.05,
                maximum_iterations=120,
                residual_threshold=1e-5,
            )
        )

        (
            reference_position,
            reference_quaternion,
        ) = controller.get_current_ee_pose()

        print()
        print(
            "reference_position:",
            np.round(
                reference_position,
                6,
            ),
        )

        print(
            "reference_quaternion:",
            np.round(
                reference_quaternion,
                6,
            ),
        )

        print(
            "ee_link:",
            controller.ee_link,
        )

        print(
            "arm_joint_angles:",
            np.round(
                controller
                .get_current_arm_joint_angles(),
                6,
            ),
        )

        target_marker_id = (
            create_target_marker(
                reference_position
            )
        )

        z_rotation = (
            axis_angle_quaternion(
                axis=np.array(
                    [0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
                angle_degrees=(
                    ROTATION_ANGLE_DEGREES
                ),
            )
        )

        rotated_quaternion = (
            quaternion_multiply(
                z_rotation,
                reference_quaternion,
            )
        )

        waypoints = [
            (
                "Move robot +X",
                reference_position
                + np.array(
                    [
                        TRANSLATION_DISTANCE,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
                reference_quaternion,
            ),
            (
                "Return from +X",
                reference_position,
                reference_quaternion,
            ),
            (
                "Move robot +Y",
                reference_position
                + np.array(
                    [
                        0.0,
                        TRANSLATION_DISTANCE,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
                reference_quaternion,
            ),
            (
                "Return from +Y",
                reference_position,
                reference_quaternion,
            ),
            (
                "Move robot +Z",
                reference_position
                + np.array(
                    [
                        0.0,
                        0.0,
                        TRANSLATION_DISTANCE,
                    ],
                    dtype=np.float64,
                ),
                reference_quaternion,
            ),
            (
                "Return from +Z",
                reference_position,
                reference_quaternion,
            ),
            (
                "Rotate around robot Z",
                reference_position,
                rotated_quaternion,
            ),
            (
                "Return orientation",
                reference_position,
                reference_quaternion,
            ),
        ]

        total_accepted = 0
        total_rejected = 0

        segment_start_position = (
            reference_position.copy()
        )

        segment_start_quaternion = (
            reference_quaternion.copy()
        )

        for (
            segment_name,
            target_position,
            target_quaternion,
        ) in waypoints:
            (
                accepted_count,
                rejected_count,
            ) = run_segment(
                controller=controller,
                simulation=simulation,
                marker_id=target_marker_id,
                segment_name=segment_name,
                start_position=(
                    segment_start_position
                ),
                start_quaternion=(
                    segment_start_quaternion
                ),
                target_position=(
                    target_position
                ),
                target_quaternion=(
                    target_quaternion
                ),
            )

            total_accepted += accepted_count
            total_rejected += rejected_count

            (
                segment_start_position,
                segment_start_quaternion,
            ) = controller.get_current_ee_pose()

        final_position, final_quaternion = (
            controller.get_current_ee_pose()
        )

        print()
        print("=" * 80)
        print("IK test completed")
        print("=" * 80)

        print(
            "total_accepted:",
            total_accepted,
        )

        print(
            "total_rejected:",
            total_rejected,
        )

        print(
            "final_position:",
            np.round(
                final_position,
                6,
            ),
        )

        print(
            "final_quaternion:",
            np.round(
                final_quaternion,
                6,
            ),
        )

        print(
            "Close the PyBullet window "
            "or press Ctrl+C to exit."
        )

        while p.isConnected():
            step_simulation(
                simulation
            )

    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )

    finally:
        environment.close()


if __name__ == "__main__":
    main()