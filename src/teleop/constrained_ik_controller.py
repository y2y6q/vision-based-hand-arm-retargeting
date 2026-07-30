from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p


@dataclass
class IKResult:
    valid: bool
    arm_joint_targets: np.ndarray
    position_error: float
    orientation_error_degrees: float
    maximum_joint_step: float
    soft_limit_clipped: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "arm_joint_targets": self.arm_joint_targets.copy(),
            "position_error": float(
                self.position_error
            ),
            "orientation_error_degrees": float(
                self.orientation_error_degrees
            ),
            "maximum_joint_step": float(
                self.maximum_joint_step
            ),
            "soft_limit_clipped": bool(
                self.soft_limit_clipped
            ),
            "message": self.message,
        }


def normalize_quaternion_xyzw(
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
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

    quaternion = quaternion / norm

    if quaternion[3] < 0.0:
        quaternion = -quaternion

    return quaternion


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = normalize_quaternion_xyzw(
        quaternion
    )

    matrix = p.getMatrixFromQuaternion(
        quaternion.tolist()
    )

    return np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(3, 3)


def orientation_error_degrees(
    target_quaternion: np.ndarray,
    actual_quaternion: np.ndarray,
) -> float:
    target_rotation = (
        quaternion_to_rotation_matrix(
            target_quaternion
        )
    )

    actual_rotation = (
        quaternion_to_rotation_matrix(
            actual_quaternion
        )
    )

    relative_rotation = (
        target_rotation
        @ actual_rotation.T
    )

    cosine_angle = float(
        np.clip(
            (
                np.trace(
                    relative_rotation
                )
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    return float(
        math.degrees(
            math.acos(
                cosine_angle
            )
        )
    )


class ConstrainedIKController:
    def __init__(
        self,
        robot,
        maximum_joint_step: float = 0.15,
        maximum_position_error: float = 0.02,
        maximum_orientation_error_degrees: float = 10.0,
        soft_limit_margin: float = 0.05,
        joint_damping: float = 0.05,
        maximum_iterations: int = 120,
        residual_threshold: float = 1e-5,
        maximum_joint_velocity: float = 1.00,
        maximum_joint_acceleration: float = 5.00,
        maximum_candidate_joint_jump: float = 0.35,
        continuity_resync_threshold: float = 0.60,
    ) -> None:
        self.robot = robot

        self.maximum_joint_step = float(
            maximum_joint_step
        )

        self.maximum_position_error = float(
            maximum_position_error
        )

        self.maximum_orientation_error_degrees = float(
            maximum_orientation_error_degrees
        )

        self.soft_limit_margin = float(
            soft_limit_margin
        )

        self.joint_damping_value = float(
            joint_damping
        )

        self.maximum_iterations = int(
            maximum_iterations
        )

        self.residual_threshold = float(
            residual_threshold
        )

        self.maximum_joint_velocity = float(
            maximum_joint_velocity
        )

        self.maximum_joint_acceleration = float(
            maximum_joint_acceleration
        )

        self.maximum_candidate_joint_jump = float(
            maximum_candidate_joint_jump
        )

        self.continuity_resync_threshold = float(
            continuity_resync_threshold
        )

        self.body_id = (
            self._get_body_unique_id()
        )

        self.ee_link = int(
            self.robot.ee_link
        )

        self.arm_joint_indices = np.asarray(
            self.robot.arm_joint_indices,
            dtype=np.int32,
        )

        self.allegro_joint_indices = np.asarray(
            self.robot.allegro_joint_indices,
            dtype=np.int32,
        )

        self.movable_joint_indices = (
            self._find_movable_joint_indices()
        )

        self.dof_position_by_joint = {
            joint_index: dof_position
            for dof_position, joint_index
            in enumerate(
                self.movable_joint_indices
            )
        }

        self.arm_dof_positions = np.asarray(
            [
                self.dof_position_by_joint[
                    int(joint_index)
                ]
                for joint_index
                in self.arm_joint_indices
            ],
            dtype=np.int32,
        )

        (
            self.lower_limits,
            self.upper_limits,
            self.joint_ranges,
        ) = self._read_movable_joint_limits()

        (
            self.arm_soft_lower_limits,
            self.arm_soft_upper_limits,
        ) = self._calculate_arm_soft_limits()

        self.joint_damping = np.full(
            len(
                self.movable_joint_indices
            ),
            self.joint_damping_value,
            dtype=np.float64,
        )

        self.previous_arm_targets = (
            self.get_current_arm_joint_angles()
        )

        self.previous_target_velocity = (
            np.zeros(
                len(
                    self.arm_joint_indices
                ),
                dtype=np.float64,
            )
        )

        self.previous_solve_timestamp = (
            time.monotonic()
        )

        print(
            "Constrained IK controller initialized."
        )

        print(
            "body_id:",
            self.body_id,
        )

        print(
            "ee_link:",
            self.ee_link,
        )

        print(
            "arm_joint_indices:",
            self.arm_joint_indices,
        )

        print(
            "movable_joint_count:",
            len(
                self.movable_joint_indices
            ),
        )

    def _get_body_unique_id(
        self,
    ) -> int:
        if hasattr(
            self.robot,
            "_get_body_unique_id",
        ):
            return int(
                self.robot._get_body_unique_id()
            )

        return int(
            self.robot.sim._bodies_idx[
                self.robot.body_name
            ]
        )

    def _find_movable_joint_indices(
        self,
    ) -> List[int]:
        movable_joint_indices = []

        for joint_index in range(
            p.getNumJoints(
                self.body_id
            )
        ):
            joint_info = p.getJointInfo(
                self.body_id,
                joint_index,
            )

            joint_type = int(
                joint_info[2]
            )

            if joint_type != p.JOINT_FIXED:
                movable_joint_indices.append(
                    joint_index
                )

        return movable_joint_indices

    def _read_movable_joint_limits(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        lower_limits = []
        upper_limits = []
        joint_ranges = []

        for joint_index in (
            self.movable_joint_indices
        ):
            joint_info = p.getJointInfo(
                self.body_id,
                joint_index,
            )

            lower_limit = float(
                joint_info[8]
            )

            upper_limit = float(
                joint_info[9]
            )

            if (
                not np.isfinite(
                    lower_limit
                )
                or not np.isfinite(
                    upper_limit
                )
                or lower_limit >= upper_limit
            ):
                lower_limit = -math.pi
                upper_limit = math.pi

            lower_limits.append(
                lower_limit
            )

            upper_limits.append(
                upper_limit
            )

            joint_ranges.append(
                upper_limit
                - lower_limit
            )

        return (
            np.asarray(
                lower_limits,
                dtype=np.float64,
            ),
            np.asarray(
                upper_limits,
                dtype=np.float64,
            ),
            np.asarray(
                joint_ranges,
                dtype=np.float64,
            ),
        )

    def _calculate_arm_soft_limits(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        arm_lower_limits = (
            self.lower_limits[
                self.arm_dof_positions
            ].copy()
        )

        arm_upper_limits = (
            self.upper_limits[
                self.arm_dof_positions
            ].copy()
        )

        for index in range(
            len(
                arm_lower_limits
            )
        ):
            available_range = (
                arm_upper_limits[index]
                - arm_lower_limits[index]
            )

            if (
                available_range
                > 2.0
                * self.soft_limit_margin
            ):
                arm_lower_limits[index] += (
                    self.soft_limit_margin
                )

                arm_upper_limits[index] -= (
                    self.soft_limit_margin
                )

        return (
            arm_lower_limits,
            arm_upper_limits,
        )

    def get_current_arm_joint_angles(
        self,
    ) -> np.ndarray:
        return np.asarray(
            [
                p.getJointState(
                    self.body_id,
                    int(joint_index),
                )[0]
                for joint_index
                in self.arm_joint_indices
            ],
            dtype=np.float64,
        )

    def get_current_allegro_joint_angles(
        self,
    ) -> np.ndarray:
        return np.asarray(
            [
                p.getJointState(
                    self.body_id,
                    int(joint_index),
                )[0]
                for joint_index
                in self.allegro_joint_indices
            ],
            dtype=np.float64,
        )

    def get_current_movable_joint_angles(
        self,
    ) -> np.ndarray:
        return np.asarray(
            [
                p.getJointState(
                    self.body_id,
                    int(joint_index),
                )[0]
                for joint_index
                in self.movable_joint_indices
            ],
            dtype=np.float64,
        )

    def get_current_ee_pose(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        link_state = p.getLinkState(
            self.body_id,
            self.ee_link,
            computeForwardKinematics=True,
        )

        position = np.asarray(
            link_state[4],
            dtype=np.float64,
        )

        quaternion = (
            normalize_quaternion_xyzw(
                np.asarray(
                    link_state[5],
                    dtype=np.float64,
                )
            )
        )

        return (
            position,
            quaternion,
        )

    def reset_continuity_reference(
        self,
    ) -> None:
        self.previous_arm_targets = (
            self.get_current_arm_joint_angles()
        )

        self.previous_target_velocity = (
            np.zeros(
                len(
                    self.arm_joint_indices
                ),
                dtype=np.float64,
            )
        )

        self.previous_solve_timestamp = (
            time.monotonic()
        )

        print(
            "IK continuity reference reset."
        )

    def _synchronize_reference_if_needed(
        self,
        current_arm_angles: np.ndarray,
    ) -> bool:
        if (
            self.previous_arm_targets.shape
            != current_arm_angles.shape
            or not np.all(
                np.isfinite(
                    self.previous_arm_targets
                )
            )
        ):
            self.reset_continuity_reference()

            return True

        difference = np.abs(
            current_arm_angles
            - self.previous_arm_targets
        )

        if (
            float(
                np.max(
                    difference
                )
            )
            > self.continuity_resync_threshold
        ):
            self.previous_arm_targets = (
                current_arm_angles.copy()
            )

            self.previous_target_velocity = (
                np.zeros_like(
                    current_arm_angles
                )
            )

            self.previous_solve_timestamp = (
                time.monotonic()
            )

            return True

        return False

    def _calculate_delta_time(
        self,
    ) -> float:
        current_time = (
            time.monotonic()
        )

        delta_time = (
            current_time
            - self.previous_solve_timestamp
        )

        self.previous_solve_timestamp = (
            current_time
        )

        return float(
            np.clip(
                delta_time,
                1.0 / 240.0,
                0.10,
            )
        )

    def _build_rest_poses(
        self,
        current_movable_angles: np.ndarray,
    ) -> np.ndarray:
        rest_poses = (
            current_movable_angles.copy()
        )

        rest_poses[
            self.arm_dof_positions
        ] = self.previous_arm_targets

        return rest_poses

    def _calculate_candidate_fk(
        self,
        arm_joint_targets: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        saved_joint_states = []

        for joint_index in (
            self.arm_joint_indices
        ):
            joint_state = p.getJointState(
                self.body_id,
                int(joint_index),
            )

            saved_joint_states.append(
                (
                    int(
                        joint_index
                    ),
                    float(
                        joint_state[0]
                    ),
                    float(
                        joint_state[1]
                    ),
                )
            )

        try:
            for joint_index, joint_target in zip(
                self.arm_joint_indices,
                arm_joint_targets,
            ):
                p.resetJointState(
                    self.body_id,
                    int(
                        joint_index
                    ),
                    targetValue=float(
                        joint_target
                    ),
                    targetVelocity=0.0,
                )

            link_state = p.getLinkState(
                self.body_id,
                self.ee_link,
                computeForwardKinematics=True,
            )

            position = np.asarray(
                link_state[4],
                dtype=np.float64,
            )

            quaternion = (
                normalize_quaternion_xyzw(
                    np.asarray(
                        link_state[5],
                        dtype=np.float64,
                    )
                )
            )

        finally:
            for (
                joint_index,
                joint_position,
                joint_velocity,
            ) in saved_joint_states:
                p.resetJointState(
                    self.body_id,
                    joint_index,
                    targetValue=(
                        joint_position
                    ),
                    targetVelocity=(
                        joint_velocity
                    ),
                )

        return (
            position,
            quaternion,
        )

    def _reject(
        self,
        current_arm_angles: np.ndarray,
        message: str,
        position_error: float = float(
            "inf"
        ),
        orientation_error: float = float(
            "inf"
        ),
        maximum_joint_step: float = 0.0,
        soft_limit_clipped: bool = False,
    ) -> IKResult:
        self.previous_target_velocity *= (
            0.5
        )

        return IKResult(
            valid=False,
            arm_joint_targets=(
                current_arm_angles.copy()
            ),
            position_error=float(
                position_error
            ),
            orientation_error_degrees=float(
                orientation_error
            ),
            maximum_joint_step=float(
                maximum_joint_step
            ),
            soft_limit_clipped=bool(
                soft_limit_clipped
            ),
            message=message,
        )

    def _limit_target_motion(
        self,
        candidate_arm_angles: np.ndarray,
        delta_time: float,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        reference_angles = (
            self.previous_arm_targets
        )

        desired_delta = (
            candidate_arm_angles
            - reference_angles
        )

        desired_velocity = (
            desired_delta
            / delta_time
        )

        desired_velocity = np.clip(
            desired_velocity,
            -self.maximum_joint_velocity,
            self.maximum_joint_velocity,
        )

        desired_acceleration = (
            desired_velocity
            - self.previous_target_velocity
        ) / delta_time

        limited_acceleration = np.clip(
            desired_acceleration,
            -self.maximum_joint_acceleration,
            self.maximum_joint_acceleration,
        )

        limited_velocity = (
            self.previous_target_velocity
            + limited_acceleration
            * delta_time
        )

        limited_velocity = np.clip(
            limited_velocity,
            -self.maximum_joint_velocity,
            self.maximum_joint_velocity,
        )

        command_delta = (
            limited_velocity
            * delta_time
        )

        command_delta = np.clip(
            command_delta,
            -self.maximum_joint_step,
            self.maximum_joint_step,
        )

        for index in range(
            len(
                command_delta
            )
        ):
            if (
                abs(
                    desired_delta[index]
                )
                < 1e-10
            ):
                command_delta[index] = (
                    0.0
                )

            elif (
                np.sign(
                    command_delta[index]
                )
                != np.sign(
                    desired_delta[index]
                )
            ):
                command_delta[index] = (
                    0.0
                )

            elif (
                abs(
                    command_delta[index]
                )
                > abs(
                    desired_delta[index]
                )
            ):
                command_delta[index] = (
                    desired_delta[index]
                )

        command_targets = (
            reference_angles
            + command_delta
        )

        command_targets = np.clip(
            command_targets,
            self.arm_soft_lower_limits,
            self.arm_soft_upper_limits,
        )

        actual_velocity = (
            command_targets
            - reference_angles
        ) / delta_time

        return (
            command_targets,
            actual_velocity,
        )

    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion_xyzw: np.ndarray,
    ) -> IKResult:
        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        ).reshape(3)

        target_quaternion = (
            normalize_quaternion_xyzw(
                target_quaternion_xyzw
            )
        )

        current_arm_angles = (
            self.get_current_arm_joint_angles()
        )

        current_movable_angles = (
            self.get_current_movable_joint_angles()
        )

        reference_resynchronized = (
            self._synchronize_reference_if_needed(
                current_arm_angles
            )
        )

        delta_time = (
            self._calculate_delta_time()
        )

        if (
            not np.all(
                np.isfinite(
                    target_position
                )
            )
            or not np.all(
                np.isfinite(
                    target_quaternion
                )
            )
        ):
            return self._reject(
                current_arm_angles,
                (
                    "Target pose contains "
                    "invalid values."
                ),
            )

        rest_poses = (
            self._build_rest_poses(
                current_movable_angles
            )
        )

        ik_solution = (
            p.calculateInverseKinematics(
                bodyUniqueId=(
                    self.body_id
                ),
                endEffectorLinkIndex=(
                    self.ee_link
                ),
                targetPosition=(
                    target_position.tolist()
                ),
                targetOrientation=(
                    target_quaternion.tolist()
                ),
                lowerLimits=(
                    self.lower_limits.tolist()
                ),
                upperLimits=(
                    self.upper_limits.tolist()
                ),
                jointRanges=(
                    self.joint_ranges.tolist()
                ),
                restPoses=(
                    rest_poses.tolist()
                ),
                jointDamping=(
                    self.joint_damping.tolist()
                ),
                solver=p.IK_DLS,
                maxNumIterations=(
                    self.maximum_iterations
                ),
                residualThreshold=(
                    self.residual_threshold
                ),
            )
        )

        ik_solution = np.asarray(
            ik_solution,
            dtype=np.float64,
        )

        if (
            ik_solution.size
            < len(
                self.movable_joint_indices
            )
        ):
            return self._reject(
                current_arm_angles,
                (
                    "IK solution has an "
                    "unexpected size."
                ),
            )

        candidate_arm_angles = (
            ik_solution[
                self.arm_dof_positions
            ].copy()
        )

        if not np.all(
            np.isfinite(
                candidate_arm_angles
            )
        ):
            return self._reject(
                current_arm_angles,
                (
                    "IK solution contains "
                    "invalid joint values."
                ),
            )

        clipped_arm_angles = np.clip(
            candidate_arm_angles,
            self.arm_soft_lower_limits,
            self.arm_soft_upper_limits,
        )

        soft_limit_clipped = (
            not np.allclose(
                clipped_arm_angles,
                candidate_arm_angles,
                atol=1e-9,
            )
        )

        raw_joint_steps = np.abs(
            clipped_arm_angles
            - self.previous_arm_targets
        )

        raw_maximum_joint_step = float(
            np.max(
                raw_joint_steps
            )
        )

        if (
            raw_maximum_joint_step
            > self.maximum_candidate_joint_jump
        ):
            return self._reject(
                current_arm_angles,
                (
                    "IK rejected: candidate "
                    "branch jump."
                ),
                maximum_joint_step=(
                    raw_maximum_joint_step
                ),
                soft_limit_clipped=(
                    soft_limit_clipped
                ),
            )

        (
            predicted_position,
            predicted_quaternion,
        ) = self._calculate_candidate_fk(
            clipped_arm_angles
        )

        position_error = float(
            np.linalg.norm(
                predicted_position
                - target_position
            )
        )

        rotation_error = (
            orientation_error_degrees(
                target_quaternion,
                predicted_quaternion,
            )
        )

        if (
            position_error
            > self.maximum_position_error
        ):
            return self._reject(
                current_arm_angles,
                (
                    "IK rejected: position "
                    "residual is too large."
                ),
                position_error=(
                    position_error
                ),
                orientation_error=(
                    rotation_error
                ),
                maximum_joint_step=(
                    raw_maximum_joint_step
                ),
                soft_limit_clipped=(
                    soft_limit_clipped
                ),
            )

        if (
            rotation_error
            > self.maximum_orientation_error_degrees
        ):
            return self._reject(
                current_arm_angles,
                (
                    "IK rejected: orientation "
                    "residual is too large."
                ),
                position_error=(
                    position_error
                ),
                orientation_error=(
                    rotation_error
                ),
                maximum_joint_step=(
                    raw_maximum_joint_step
                ),
                soft_limit_clipped=(
                    soft_limit_clipped
                ),
            )

        (
            command_arm_targets,
            command_velocity,
        ) = self._limit_target_motion(
            clipped_arm_angles,
            delta_time,
        )

        command_joint_steps = np.abs(
            command_arm_targets
            - self.previous_arm_targets
        )

        maximum_command_joint_step = float(
            np.max(
                command_joint_steps
            )
        )

        self.previous_arm_targets = (
            command_arm_targets.copy()
        )

        self.previous_target_velocity = (
            command_velocity.copy()
        )

        message = (
            "IK solution accepted."
        )

        if reference_resynchronized:
            message = (
                "IK solution accepted after "
                "continuity resynchronization."
            )

        return IKResult(
            valid=True,
            arm_joint_targets=(
                command_arm_targets
            ),
            position_error=(
                position_error
            ),
            orientation_error_degrees=(
                rotation_error
            ),
            maximum_joint_step=(
                maximum_command_joint_step
            ),
            soft_limit_clipped=(
                soft_limit_clipped
            ),
            message=message,
        )

    def apply(
        self,
        result: IKResult,
    ) -> bool:
        if not result.valid:
            return False

        current_allegro_angles = (
            self.get_current_allegro_joint_angles()
        )

        all_joint_targets = np.concatenate(
            [
                result.arm_joint_targets,
                current_allegro_angles,
            ]
        )

        self.robot.control_joints(
            target_angles=(
                all_joint_targets
            )
        )

        return True

    def solve_and_apply(
        self,
        target_position: np.ndarray,
        target_quaternion_xyzw: np.ndarray,
    ) -> IKResult:
        result = self.solve(
            target_position=(
                target_position
            ),
            target_quaternion_xyzw=(
                target_quaternion_xyzw
            ),
        )

        if result.valid:
            self.apply(
                result
            )

        return result