import time
from pathlib import Path
from typing import Dict

import numpy as np
import pybullet as p
import pybullet_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "vendor"
    / "panda-gym"
    / "panda_gym"
    / "assets"
    / "robots"
    / "panda_allegro"
    / "panda_allegro.urdf"
)

FRAME_NAMES = [
    "palm",
    "allegro_tcp",
    "grasp_tcp",
]

AXIS_LENGTH = 0.055
AXIS_WIDTH = 3.0


def find_link_indices(
    robot_id: int,
) -> Dict[str, int]:
    links = {}

    for joint_index in range(
        p.getNumJoints(robot_id)
    ):
        info = p.getJointInfo(
            robot_id,
            joint_index,
        )

        link_name = info[12].decode(
            "utf-8"
        )

        links[link_name] = joint_index

    return links


def get_link_pose(
    robot_id: int,
    link_index: int,
):
    state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True,
    )

    position = np.asarray(
        state[4],
        dtype=np.float64,
    )

    orientation = np.asarray(
        state[5],
        dtype=np.float64,
    )

    return position, orientation


def transform_point(
    position,
    orientation,
    local_point,
):
    world_point, _ = p.multiplyTransforms(
        np.asarray(
            position,
            dtype=np.float64,
        ).tolist(),
        np.asarray(
            orientation,
            dtype=np.float64,
        ).tolist(),
        np.asarray(
            local_point,
            dtype=np.float64,
        ).tolist(),
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )

    return np.asarray(
        world_point,
        dtype=np.float64,
    )


def draw_frame(
    position,
    orientation,
    label: str,
):
    origin = np.asarray(
        position,
        dtype=np.float64,
    )

    x_end = transform_point(
        position,
        orientation,
        [
            AXIS_LENGTH,
            0.0,
            0.0,
        ],
    )

    y_end = transform_point(
        position,
        orientation,
        [
            0.0,
            AXIS_LENGTH,
            0.0,
        ],
    )

    z_end = transform_point(
        position,
        orientation,
        [
            0.0,
            0.0,
            AXIS_LENGTH,
        ],
    )

    # X 轴：红色
    p.addUserDebugLine(
        origin,
        x_end,
        [
            1.0,
            0.0,
            0.0,
        ],
        AXIS_WIDTH,
        0,
    )

    # Y 轴：绿色
    p.addUserDebugLine(
        origin,
        y_end,
        [
            0.0,
            1.0,
            0.0,
        ],
        AXIS_WIDTH,
        0,
    )

    # Z 轴：蓝色
    p.addUserDebugLine(
        origin,
        z_end,
        [
            0.0,
            0.0,
            1.0,
        ],
        AXIS_WIDTH,
        0,
    )

    p.addUserDebugText(
        label,
        origin
        + np.array(
            [
                0.0,
                0.0,
                0.012,
            ],
            dtype=np.float64,
        ),
        [
            1.0,
            1.0,
            0.0,
        ],
        1.35,
        0,
    )


def add_grasp_marker(
    position,
):
    visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.012,
        rgbaColor=[
            1.0,
            0.0,
            1.0,
            0.9,
        ],
    )

    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_id,
        basePosition=np.asarray(
            position,
            dtype=np.float64,
        ).tolist(),
    )


def main():
    print("=" * 80)
    print("Grasp TCP debug")
    print("=" * 80)
    print("URDF:", URDF_PATH)

    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"URDF does not exist: {URDF_PATH}"
        )

    physics_client = p.connect(
        p.GUI
    )

    if physics_client < 0:
        raise RuntimeError(
            "Cannot connect to PyBullet GUI."
        )

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.resetSimulation()

        p.setGravity(
            0.0,
            0.0,
            -9.81,
        )

        p.loadURDF(
            "plane.urdf"
        )

        robot_id = p.loadURDF(
            str(URDF_PATH),
            basePosition=[
                0.0,
                0.0,
                0.0,
            ],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        link_indices = find_link_indices(
            robot_id
        )

        missing = [
            name
            for name in FRAME_NAMES
            if name not in link_indices
        ]

        if missing:
            raise RuntimeError(
                f"Missing links: {missing}. "
                f"Available: "
                f"{list(link_indices.keys())}"
            )

        poses = {}

        for name in FRAME_NAMES:
            poses[name] = get_link_pose(
                robot_id,
                link_indices[name],
            )

            draw_frame(
                poses[name][0],
                poses[name][1],
                name,
            )

        grasp_position, _ = poses[
            "grasp_tcp"
        ]

        palm_position, _ = poses[
            "palm"
        ]

        add_grasp_marker(
            grasp_position
        )

        # palm 到 grasp_tcp 的紫色连线
        p.addUserDebugLine(
            palm_position,
            grasp_position,
            [
                1.0,
                0.0,
                1.0,
            ],
            4.0,
            0,
        )

        print()
        print("=" * 80)
        print("Frame information")
        print("=" * 80)

        for name in FRAME_NAMES:
            position, orientation = poses[
                name
            ]

            print(
                f"{name}_index: "
                f"{link_indices[name]}"
            )

            print(
                f"{name}_position: "
                f"{np.round(position, 6)}"
            )

            print(
                f"{name}_quaternion: "
                f"{np.round(orientation, 6)}"
            )

        palm_to_grasp_world = (
            grasp_position
            - palm_position
        )

        movable_joint_count = sum(
            p.getJointInfo(
                robot_id,
                joint_index,
            )[2]
            != p.JOINT_FIXED
            for joint_index
            in range(
                p.getNumJoints(robot_id)
            )
        )

        print(
            "palm_to_grasp_world:",
            np.round(
                palm_to_grasp_world,
                6,
            ),
        )

        print(
            "Movable joint count:",
            movable_joint_count,
        )

        print()
        print(
            "Close the PyBullet window "
            "or press Ctrl+C to exit."
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=1.05,
            cameraYaw=55.0,
            cameraPitch=-22.0,
            cameraTargetPosition=[
                0.0,
                0.0,
                0.72,
            ],
        )

        while p.isConnected():
            p.stepSimulation()
            time.sleep(
                1.0 / 240.0
            )

    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )

    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
