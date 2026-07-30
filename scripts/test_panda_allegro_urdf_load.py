import time
from pathlib import Path

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


def main():
    print("=" * 80)
    print("Panda + upright Allegro URDF load test")
    print("=" * 80)
    print("URDF:", URDF_PATH)

    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"URDF does not exist: {URDF_PATH}"
        )

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError("Cannot connect to PyBullet GUI.")

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=physics_client,
        )

        p.resetSimulation(
            physicsClientId=physics_client,
        )

        p.setGravity(
            0.0,
            0.0,
            -9.81,
            physicsClientId=physics_client,
        )

        p.loadURDF(
            "plane.urdf",
            physicsClientId=physics_client,
        )

        robot_id = p.loadURDF(
            fileName=str(URDF_PATH),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler(
                [0.0, 0.0, 0.0]
            ),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=physics_client,
        )

        number_of_joints = p.getNumJoints(
            robot_id,
            physicsClientId=physics_client,
        )

        movable_joints = []

        print()
        print("URDF loaded successfully.")
        print("robot_id:", robot_id)
        print("number_of_joints:", number_of_joints)
        print()
        print("Joint list:")

        for joint_index in range(number_of_joints):
            joint_info = p.getJointInfo(
                robot_id,
                joint_index,
                physicsClientId=physics_client,
            )

            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            child_link_name = joint_info[12].decode("utf-8")

            if joint_type != p.JOINT_FIXED:
                movable_joints.append(joint_index)

            print(
                f"joint_index={joint_index:02d}, "
                f"name={joint_name}, "
                f"type={joint_type}, "
                f"child_link={child_link_name}"
            )

        print()
        print("Movable joints:", movable_joints)
        print("Movable joint count:", len(movable_joints))

        p.resetDebugVisualizerCamera(
            cameraDistance=1.45,
            cameraYaw=45.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[0.0, 0.0, 0.55],
            physicsClientId=physics_client,
        )

        print()
        print("=" * 80)
        print("The PyBullet window will remain open.")
        print("Close the PyBullet window or press Ctrl+C to exit.")
        print("=" * 80)

        while p.isConnected(
            physicsClientId=physics_client
        ):
            p.stepSimulation(
                physicsClientId=physics_client,
            )

            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as error:
        print()
        print("=" * 80)
        print("URDF LOAD FAILED")
        print("=" * 80)
        print(type(error).__name__ + ":", error)

        input(
            "\nPress Enter to close the PyBullet connection..."
        )

        raise

    finally:
        if p.isConnected(
            physicsClientId=physics_client
        ):
            p.disconnect(
                physicsClientId=physics_client,
            )


if __name__ == "__main__":
    main()
