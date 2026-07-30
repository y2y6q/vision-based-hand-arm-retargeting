from pathlib import Path
from typing import List, Optional

import numpy as np
import pybullet as p
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


PANDA_ALLEGRO_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "panda_allegro"
    / "panda_allegro.urdf"
)


class PandaAllegro(PyBulletRobot):
    """Panda arm with Allegro Hand robot in PyBullet.

    Action design:

        control_type == "ee":
            action[:3]    -> Panda end-effector displacement
            action[3:19]  -> Allegro 16 joint incremental control

        control_type == "joints":
            action[:7]    -> Panda arm joint incremental control
            action[7:23]  -> Allegro 16 joint incremental control
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = (
            base_position
            if base_position is not None
            else np.zeros(3)
        )

        self.block_gripper = block_gripper
        self.control_type = control_type

        # Panda arm 7 movable joints.
        self.arm_joint_indices = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            dtype=np.int32,
        )

        # Combined Panda + Allegro URDF PyBullet indices.
        #
        # 11 = allegro_tcp_joint, fixed
        # 12 = grasp_tcp_joint, fixed
        #
        # Finger movable joints:
        #
        # Index:
        #   joint_0.0  -> 13
        #   joint_1.0  -> 14
        #   joint_2.0  -> 15
        #   joint_3.0  -> 16
        #
        # Middle:
        #   joint_4.0  -> 18
        #   joint_5.0  -> 19
        #   joint_6.0  -> 20
        #   joint_7.0  -> 21
        #
        # Ring:
        #   joint_8.0  -> 23
        #   joint_9.0  -> 24
        #   joint_10.0 -> 25
        #   joint_11.0 -> 26
        #
        # Thumb:
        #   joint_12.0 -> 28
        #   joint_13.0 -> 29
        #   joint_14.0 -> 30
        #   joint_15.0 -> 31
        self.allegro_joint_indices = np.array(
            [
                13,
                14,
                15,
                16,

                18,
                19,
                20,
                21,

                23,
                24,
                25,
                26,

                28,
                29,
                30,
                31,
            ],
            dtype=np.int32,
        )

        self.allegro_joint_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",

            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",

            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",

            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
        ]

        self.allegro_lower_limits = np.array(
            [
                -0.470,
                -0.196,
                -0.174,
                -0.227,

                -0.470,
                -0.196,
                -0.174,
                -0.227,

                -0.470,
                -0.196,
                -0.174,
                -0.227,

                0.263,
                -0.105,
                -0.189,
                -0.162,
            ],
            dtype=np.float32,
        )

        self.allegro_upper_limits = np.array(
            [
                0.470,
                1.610,
                1.709,
                1.618,

                0.470,
                1.610,
                1.709,
                1.618,

                0.470,
                1.610,
                1.709,
                1.618,

                1.396,
                1.163,
                1.644,
                1.719,
            ],
            dtype=np.float32,
        )

        self.all_control_joint_indices = np.concatenate(
            [
                self.arm_joint_indices,
                self.allegro_joint_indices,
            ]
        )

        # Panda arm neutral pose.
        self.arm_neutral_joint_values = np.array(
            [
                0.00,
                0.41,
                0.00,
                -1.85,
                0.00,
                2.26,
                0.79,
            ],
            dtype=np.float32,
        )

        # Allegro neutral pose.
        self.allegro_neutral_joint_values = np.clip(
            np.zeros(
                16,
                dtype=np.float32,
            ),
            self.allegro_lower_limits,
            self.allegro_upper_limits,
        )

        self.neutral_joint_values = np.concatenate(
            [
                self.arm_neutral_joint_values,
                self.allegro_neutral_joint_values,
            ]
        )

        if self.control_type == "ee":
            n_action = 3 + 16

        elif self.control_type == "joints":
            n_action = 7 + 16

        else:
            raise ValueError(
                f"Unknown control_type: "
                f"{self.control_type}"
            )

        action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(n_action,),
            dtype=np.float32,
        )

        arm_joint_forces = np.array(
            [
                87.0,
                87.0,
                87.0,
                87.0,
                12.0,
                120.0,
                120.0,
            ],
            dtype=np.float32,
        )

        allegro_joint_forces = (
            np.ones(
                16,
                dtype=np.float32,
            )
            * 1.0
        )

        joint_forces = np.concatenate(
            [
                arm_joint_forces,
                allegro_joint_forces,
            ]
        )

        super().__init__(
            sim,
            body_name="panda",
            file_name=str(
                PANDA_ALLEGRO_URDF
            ),
            base_position=base_position,
            action_space=action_space,
            joint_indices=(
                self.all_control_joint_indices
            ),
            joint_forces=joint_forces,
        )

        self._validate_allegro_joint_mapping()

        self.wrist_link = self._find_link_index(
            [
                "allegro_tcp",
                "wrist",
                "base_link",
            ]
        )

        self.grasp_link = self._find_link_index(
            [
                "grasp_tcp",
                "allegro_tcp",
                "wrist",
                "base_link",
            ]
        )

        self.ee_link = self.grasp_link

        print(
            "PandaAllegro control links: "
            f"wrist_link={self.wrist_link}, "
            f"grasp_link={self.grasp_link}, "
            f"ee_link={self.ee_link}"
        )

    def _get_body_unique_id(
        self,
    ) -> int:
        return self.sim._bodies_idx[
            self.body_name
        ]

    def _validate_allegro_joint_mapping(
        self,
    ) -> None:
        body_id = self._get_body_unique_id()

        actual_joint_names = []

        for joint_index in (
            self.allegro_joint_indices
        ):
            joint_info = p.getJointInfo(
                body_id,
                int(joint_index),
            )

            joint_name = (
                joint_info[1]
                .decode("utf-8")
            )

            joint_type = int(
                joint_info[2]
            )

            if joint_type == p.JOINT_FIXED:
                raise RuntimeError(
                    "Allegro control mapping "
                    "contains a fixed joint: "
                    f"index={joint_index}, "
                    f"name={joint_name}"
                )

            actual_joint_names.append(
                joint_name
            )

        if (
            actual_joint_names
            != self.allegro_joint_names
        ):
            raise RuntimeError(
                "Allegro joint mapping mismatch.\n"
                f"Expected: "
                f"{self.allegro_joint_names}\n"
                f"Actual: "
                f"{actual_joint_names}\n"
                f"Indices: "
                f"{self.allegro_joint_indices.tolist()}"
            )

        print(
            "Allegro joint mapping verified:"
        )

        for (
            joint_index,
            joint_name,
        ) in zip(
            self.allegro_joint_indices,
            actual_joint_names,
        ):
            print(
                f"  index={int(joint_index):2d} "
                f"name={joint_name}"
            )

    def _find_link_index(
        self,
        candidate_link_names: List[str],
    ) -> int:
        body_id = self._get_body_unique_id()

        available_links = {}

        for joint_index in range(
            p.getNumJoints(
                body_id
            )
        ):
            info = p.getJointInfo(
                body_id,
                joint_index,
            )

            link_name = (
                info[12]
                .decode("utf-8")
            )

            available_links[
                link_name
            ] = joint_index

        for name in candidate_link_names:
            if name in available_links:
                print(
                    "PandaAllegro ee_link selected: "
                    f"{name}, "
                    f"index={available_links[name]}"
                )

                return available_links[
                    name
                ]

        raise RuntimeError(
            "Cannot find any ee link from "
            f"{candidate_link_names}. "
            "Available links: "
            f"{list(available_links.keys())}"
        )

    def set_action(
        self,
        action: np.ndarray,
    ) -> None:
        action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        if self.control_type == "ee":
            arm_action = action[:3]
            hand_action = action[3:]

            target_arm_angles = (
                self.ee_displacement_to_target_arm_angles(
                    arm_action
                )
            )

        else:
            arm_action = action[:7]
            hand_action = action[7:]

            target_arm_angles = (
                self.arm_joint_ctrl_to_target_arm_angles(
                    arm_action
                )
            )

        target_allegro_angles = (
            self.allegro_ctrl_to_target_joint_angles(
                hand_action
            )
        )

        target_angles = np.concatenate(
            [
                target_arm_angles,
                target_allegro_angles,
            ]
        )

        self.control_joints(
            target_angles=target_angles
        )

    def ee_displacement_to_target_arm_angles(
        self,
        ee_displacement: np.ndarray,
    ) -> np.ndarray:
        ee_displacement = (
            np.asarray(
                ee_displacement,
                dtype=np.float32,
            )[:3]
            * 0.05
        )

        ee_position = np.asarray(
            self.get_ee_position(),
            dtype=np.float32,
        )

        target_ee_position = (
            ee_position
            + ee_displacement
        )

        target_ee_position[2] = np.max(
            (
                0.0,
                target_ee_position[2],
            )
        )

        target_arm_angles = (
            self.inverse_kinematics(
                link=self.ee_link,
                position=target_ee_position,
                orientation=np.array(
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            )
        )

        return np.asarray(
            target_arm_angles[:7],
            dtype=np.float32,
        )

    def arm_joint_ctrl_to_target_arm_angles(
        self,
        arm_joint_ctrl: np.ndarray,
    ) -> np.ndarray:
        arm_joint_ctrl = (
            np.asarray(
                arm_joint_ctrl,
                dtype=np.float32,
            )
            * 0.05
        )

        current_arm_joint_angles = np.array(
            [
                self.get_joint_angle(
                    joint=int(index)
                )
                for index
                in self.arm_joint_indices
            ],
            dtype=np.float32,
        )

        target_arm_angles = (
            current_arm_joint_angles
            + arm_joint_ctrl
        )

        return target_arm_angles

    def allegro_ctrl_to_target_joint_angles(
        self,
        hand_action: np.ndarray,
    ) -> np.ndarray:
        hand_action = np.asarray(
            hand_action,
            dtype=np.float32,
        )

        hand_delta = (
            hand_action
            * 0.05
        )

        current_allegro_angles = (
            self.get_allegro_joint_positions()
        )

        target_allegro_angles = (
            current_allegro_angles
            + hand_delta
        )

        target_allegro_angles = np.clip(
            target_allegro_angles,
            self.allegro_lower_limits,
            self.allegro_upper_limits,
        )

        return target_allegro_angles

    def get_obs(
        self,
    ) -> np.ndarray:
        ee_position = np.asarray(
            self.get_ee_position(),
            dtype=np.float32,
        )

        ee_velocity = np.asarray(
            self.get_ee_velocity(),
            dtype=np.float32,
        )

        allegro_joint_positions = (
            self.get_allegro_joint_positions()
        )

        allegro_joint_velocities = (
            self.get_allegro_joint_velocities()
        )

        observation = np.concatenate(
            [
                ee_position,
                ee_velocity,
                allegro_joint_positions,
                allegro_joint_velocities,
            ]
        )

        return observation

    def reset(
        self,
    ) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(
        self,
    ) -> None:
        self.set_joint_angles(
            self.neutral_joint_values
        )

    def get_allegro_joint_positions(
        self,
    ) -> np.ndarray:
        return np.array(
            [
                self.get_joint_angle(
                    joint=int(index)
                )
                for index
                in self.allegro_joint_indices
            ],
            dtype=np.float32,
        )

    def get_allegro_joint_velocities(
        self,
    ) -> np.ndarray:
        return np.array(
            [
                self.get_joint_velocity(
                    joint=int(index)
                )
                for index
                in self.allegro_joint_indices
            ],
            dtype=np.float32,
        )

    def get_ee_position(
        self,
    ) -> np.ndarray:
        return self.get_link_position(
            self.ee_link
        )

    def get_ee_velocity(
        self,
    ) -> np.ndarray:
        return self.get_link_velocity(
            self.ee_link
        )

    def get_fingers_width(
        self,
    ) -> float:
        return 0.0