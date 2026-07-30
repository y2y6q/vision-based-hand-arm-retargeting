import time

import numpy as np

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_allegro import PandaAllegro


def step_with_action(robot, sim, action, steps=240, label=""):
    print("\n" + "=" * 80)
    print(label)
    print("action:", action)
    print("=" * 80)

    for i in range(steps):
        robot.set_action(action)
        sim.step()

        if i % 60 == 0:
            ee_pos = robot.get_ee_position()
            allegro_q = robot.get_allegro_joint_positions()

            print(f"step={i}")
            print("ee_pos:", ee_pos)
            print("allegro_q[:4]:", allegro_q[:4])

        time.sleep(1.0 / 240.0)


def main():
    sim = PyBullet(render_mode="human")

    robot = PandaAllegro(
        sim=sim,
        block_gripper=False,
        base_position=None,
        control_type="ee",
    )

    robot.reset()

    print("PandaAllegro action test started.")
    print("action_space:", robot.action_space)
    print("joint_indices:", robot.joint_indices)
    print("num joint_indices:", len(robot.joint_indices))
    print("ee_link:", robot.ee_link)
    print("initial ee_pos:", robot.get_ee_position())
    print("initial allegro_q:", robot.get_allegro_joint_positions())

    zero_action = np.zeros(19, dtype=np.float32)

    # 0. 保持静止
    step_with_action(
        robot,
        sim,
        zero_action,
        steps=240,
        label="Phase 0: hold still",
    )

    # 1. 测试 Panda arm +X
    action = np.zeros(19, dtype=np.float32)
    action[0] = 0.5
    step_with_action(
        robot,
        sim,
        action,
        steps=240,
        label="Phase 1: move arm +X",
    )

    # 2. 测试 Panda arm -X
    action = np.zeros(19, dtype=np.float32)
    action[0] = -0.5
    step_with_action(
        robot,
        sim,
        action,
        steps=240,
        label="Phase 2: move arm -X",
    )

    # 3. 测试 Panda arm +Z
    action = np.zeros(19, dtype=np.float32)
    action[2] = 0.5
    step_with_action(
        robot,
        sim,
        action,
        steps=240,
        label="Phase 3: move arm +Z",
    )

    # 4. 测试 Allegro 手指弯曲
    action = np.zeros(19, dtype=np.float32)
    action[3:] = 0.7
    step_with_action(
        robot,
        sim,
        action,
        steps=360,
        label="Phase 4: close / curl Allegro fingers",
    )

    # 5. 测试 Allegro 手指反向
    action = np.zeros(19, dtype=np.float32)
    action[3:] = -0.7
    step_with_action(
        robot,
        sim,
        action,
        steps=360,
        label="Phase 5: open / reverse Allegro fingers",
    )

    # 6. arm + hand 同时控制
    action = np.zeros(19, dtype=np.float32)
    action[1] = 0.3
    action[2] = 0.2
    action[3:] = 0.5
    step_with_action(
        robot,
        sim,
        action,
        steps=360,
        label="Phase 6: move arm and curl hand together",
    )

    print("\nAction test finished.")
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