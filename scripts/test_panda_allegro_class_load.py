import time

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_allegro import PandaAllegro


def main():
    sim = PyBullet(render_mode="human")

    robot = PandaAllegro(
        sim=sim,
        block_gripper=False,
        base_position=None,
        control_type="ee",
    )

    print("PandaAllegro class loaded successfully.")
    print("body_name:", robot.body_name)
    print("action_space:", robot.action_space)
    print("joint_indices:", robot.joint_indices)
    print("num joint_indices:", len(robot.joint_indices))

    print("Close PyBullet window or press Ctrl+C to stop.")

    try:
        while True:
            sim.step()
            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        sim.close()


if __name__ == "__main__":
    main()