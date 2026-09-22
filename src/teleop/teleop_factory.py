from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

# 导入 panda_gym 会执行环境注册。
import panda_gym

from panda_gym.envs.robots.panda_allegro import (
    PANDA_ALLEGRO_URDF,
)

from src.teleop.constrained_ik_controller import (
    ConstrainedIKController,
)

from src.teleop.pose_filter import (
    PoseFilter,
)


# ============================================================
# 环境统一配置
# ============================================================

ENVIRONMENT_ID = "PandaAllegroPickAndPlace-v0"

DEFAULT_RENDER_MODE = "human"
DEFAULT_REWARD_TYPE = "sparse"
DEFAULT_CONTROL_TYPE = "ee"
DEFAULT_RENDERER = "Tiny"


# ============================================================
# IK统一配置
# ============================================================

@dataclass(frozen=True)
class IKControllerConfig:
    maximum_joint_step: float = 0.15
    maximum_position_error: float = 0.025
    maximum_orientation_error_degrees: float = 12.0

    soft_limit_margin: float = 0.05
    joint_damping: float = 0.05

    maximum_iterations: int = 120
    residual_threshold: float = 1e-5

    maximum_joint_velocity: float = 1.00
    maximum_joint_acceleration: float = 5.00

    maximum_candidate_joint_jump: float = 0.35
    continuity_resync_threshold: float = 0.60


DEFAULT_IK_CONFIG = IKControllerConfig()


# ============================================================
# PoseFilter统一配置
# ============================================================

@dataclass(frozen=True)
class PoseFilterConfig:
    position_min_cutoff: float = 1.0
    position_beta: float = 0.03
    position_derivative_cutoff: float = 1.0

    rotation_min_cutoff: float = 1.5
    rotation_beta: float = 0.05
    rotation_derivative_cutoff: float = 1.0

    # 当前版本已经关闭第二层速度和加速度裁剪。
    maximum_linear_velocity: float = 0.0
    maximum_linear_acceleration: float = 0.0
    maximum_angular_velocity: float = 0.0
    maximum_angular_acceleration: float = 0.0

    workspace_half_extent: Tuple[
        float,
        float,
        float,
    ] = (
        0.22,
        0.22,
        0.20,
    )

    minimum_z: float = 0.08

    maximum_relative_rotation_degrees: float = 100.0


DEFAULT_POSE_FILTER_CONFIG = PoseFilterConfig()


# ============================================================
# 统一创建函数
# ============================================================

def create_environment(
    render_mode: str = DEFAULT_RENDER_MODE,
    reward_type: str = DEFAULT_REWARD_TYPE,
    control_type: str = DEFAULT_CONTROL_TYPE,
    renderer: str = DEFAULT_RENDERER,
):
    """
    创建当前项目使用的 Panda + Allegro 环境。

    实时遥操作程序和工作空间分析脚本必须调用此函数，
    不再分别手写 gym.make 参数。
    """

    environment = gym.make(
        ENVIRONMENT_ID,
        render_mode=render_mode,
        reward_type=reward_type,
        control_type=control_type,
        renderer=renderer,
    )

    return environment


def create_ik_controller(
    robot,
    config: IKControllerConfig = DEFAULT_IK_CONFIG,
) -> ConstrainedIKController:
    """
    使用统一参数创建当前 ConstrainedIKController。
    """

    return ConstrainedIKController(
        robot=robot,

        maximum_joint_step=(
            config.maximum_joint_step
        ),

        maximum_position_error=(
            config.maximum_position_error
        ),

        maximum_orientation_error_degrees=(
            config.maximum_orientation_error_degrees
        ),

        soft_limit_margin=(
            config.soft_limit_margin
        ),

        joint_damping=(
            config.joint_damping
        ),

        maximum_iterations=(
            config.maximum_iterations
        ),

        residual_threshold=(
            config.residual_threshold
        ),

        maximum_joint_velocity=(
            config.maximum_joint_velocity
        ),

        maximum_joint_acceleration=(
            config.maximum_joint_acceleration
        ),

        maximum_candidate_joint_jump=(
            config.maximum_candidate_joint_jump
        ),

        continuity_resync_threshold=(
            config.continuity_resync_threshold
        ),
    )


def create_pose_filter(
    config: PoseFilterConfig = DEFAULT_POSE_FILTER_CONFIG,
) -> PoseFilter:
    """
    使用统一参数创建当前 PoseFilter。
    """

    return PoseFilter(
        position_min_cutoff=(
            config.position_min_cutoff
        ),

        position_beta=(
            config.position_beta
        ),

        position_derivative_cutoff=(
            config.position_derivative_cutoff
        ),

        rotation_min_cutoff=(
            config.rotation_min_cutoff
        ),

        rotation_beta=(
            config.rotation_beta
        ),

        rotation_derivative_cutoff=(
            config.rotation_derivative_cutoff
        ),

        maximum_linear_velocity=(
            config.maximum_linear_velocity
        ),

        maximum_linear_acceleration=(
            config.maximum_linear_acceleration
        ),

        maximum_angular_velocity=(
            config.maximum_angular_velocity
        ),

        maximum_angular_acceleration=(
            config.maximum_angular_acceleration
        ),

        workspace_half_extent=np.asarray(
            config.workspace_half_extent,
            dtype=np.float64,
        ),

        minimum_z=(
            config.minimum_z
        ),

        maximum_relative_rotation_degrees=(
            config.maximum_relative_rotation_degrees
        ),
    )


# ============================================================
# 参数记录函数
# ============================================================

def get_shared_configuration_snapshot() -> Dict[
    str,
    Any,
]:
    """
    返回可以保存到工作空间分析结果中的统一参数。

    后续每次修改参数并重新采样时，可以记录当前使用版本，
    避免不同点云结果之间无法比较。
    """

    return {
        "environment": {
            "environment_id": ENVIRONMENT_ID,
            "render_mode": DEFAULT_RENDER_MODE,
            "reward_type": DEFAULT_REWARD_TYPE,
            "control_type": DEFAULT_CONTROL_TYPE,
            "renderer": DEFAULT_RENDERER,
            "robot_urdf": str(
                PANDA_ALLEGRO_URDF
            ),
        },
        "ik": asdict(
            DEFAULT_IK_CONFIG
        ),
        "pose_filter": asdict(
            DEFAULT_POSE_FILTER_CONFIG
        ),
    }


def get_robot_configuration_snapshot(
    robot,
) -> Dict[str, Any]:
    """
    从实际加载的 PandaAllegro 对象读取机器人构型。

    不在工作空间脚本中重新填写关节索引、
    neutral pose、TCP或Allegro关节限制。
    """

    return {
        "robot_class": (
            robot.__class__.__name__
        ),

        "body_name": str(
            robot.body_name
        ),

        "control_type": str(
            robot.control_type
        ),

        "robot_urdf": str(
            PANDA_ALLEGRO_URDF
        ),

        "arm_joint_indices": (
            np.asarray(
                robot.arm_joint_indices,
                dtype=np.int32,
            ).tolist()
        ),

        "allegro_joint_indices": (
            np.asarray(
                robot.allegro_joint_indices,
                dtype=np.int32,
            ).tolist()
        ),

        "all_control_joint_indices": (
            np.asarray(
                robot.all_control_joint_indices,
                dtype=np.int32,
            ).tolist()
        ),

        "arm_neutral_joint_values": (
            np.asarray(
                robot.arm_neutral_joint_values,
                dtype=np.float64,
            ).tolist()
        ),

        "allegro_neutral_joint_values": (
            np.asarray(
                robot.allegro_neutral_joint_values,
                dtype=np.float64,
            ).tolist()
        ),

        "neutral_joint_values": (
            np.asarray(
                robot.neutral_joint_values,
                dtype=np.float64,
            ).tolist()
        ),

        "allegro_lower_limits": (
            np.asarray(
                robot.allegro_lower_limits,
                dtype=np.float64,
            ).tolist()
        ),

        "allegro_upper_limits": (
            np.asarray(
                robot.allegro_upper_limits,
                dtype=np.float64,
            ).tolist()
        ),

        "wrist_link": int(
            robot.wrist_link
        ),

        "grasp_link": int(
            robot.grasp_link
        ),

        "ee_link": int(
            robot.ee_link
        ),
    }
