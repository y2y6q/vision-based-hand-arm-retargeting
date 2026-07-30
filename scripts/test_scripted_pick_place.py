import time

import gymnasium as gym
import numpy as np
import panda_gym


def get_ee_position(observation):
    """
    获取 Panda 夹爪末端执行器位置。
    observation["observation"] 的前 3 维通常是末端执行器位置。
    """
    return observation["observation"][:3]


def force_goal_on_table(env, observation):
    """
    强制把浅绿色目标方块放到桌面上。

    原版 PandaPickAndPlace-v3 的 desired_goal 可能悬空。
    这里把 goal_position[2] 改成真实方块中心高度。
    """
    object_position = observation["achieved_goal"].copy()
    goal_position = observation["desired_goal"].copy()

    # 强制目标点落在桌面高度
    goal_position[2] = object_position[2]

    # 修改 panda-gym 内部目标
    env.unwrapped.task.goal = goal_position.copy()

    # 修改浅绿色目标 marker 的显示位置
    target_orientation = np.array([0.0, 0.0, 0.0, 1.0])

    try:
        env.unwrapped.task.sim.set_base_pose(
            "target",
            goal_position,
            target_orientation,
        )
    except Exception as e:
        print("Warning: failed to move target marker:", e)

    # 同步 observation
    observation["desired_goal"] = goal_position.copy()

    return observation, object_position, goal_position


def move_to_position(
    env,
    observation,
    target_position,
    gripper_action,
    max_steps=300,
    tolerance=0.01,
    speed_gain=3.0,
):
    """
    控制夹爪末端移动到目标位置。

    action = [dx, dy, dz, gripper]
    gripper_action = +1：张开夹爪
    gripper_action = -1：闭合夹爪
    """
    for step in range(max_steps):
        ee_position = get_ee_position(observation)
        error = target_position - ee_position
        distance = np.linalg.norm(error)

        action_xyz = speed_gain * error
        action_xyz = np.clip(action_xyz, -1.0, 1.0)

        action = np.array(
            [action_xyz[0], action_xyz[1], action_xyz[2], gripper_action],
            dtype=np.float32,
        )

        observation, reward, terminated, truncated, info = env.step(action)

        if step % 80 == 0:
            print(
                f"move step={step}, "
                f"distance={distance:.4f}, "
                f"ee={ee_position}, "
                f"target={target_position}"
            )

        time.sleep(0.01)

        if distance < tolerance:
            break

    return observation


def hold_gripper(env, observation, gripper_action, steps=100):
    """
    原地保持，同时持续控制夹爪。

    gripper_action = +1：张开
    gripper_action = -1：闭合
    """
    for _ in range(steps):
        action = np.array([0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.01)

    return observation


def run_one_pick_place_round(env, round_id):
    """
    执行一轮完整 pick-and-place：
    reset 环境 → 强制目标落桌面 → 抓取 → 移动 → 放置。
    """
    print("\n" + "=" * 70)
    print(f"Round {round_id}: reset environment and generate new target")
    print("=" * 70)

    observation, info = env.reset()

    # 强制目标点在桌面上
    observation, object_position, goal_position = force_goal_on_table(env, observation)

    print("object position:", object_position)
    print("forced table goal position:", goal_position)

    # 高度参数
    above_height = 0.15
    grasp_height = 0.01
    place_height = 0.02

    # 抓取目标点
    above_object = object_position + np.array([0.0, 0.0, above_height])
    grasp_position = object_position + np.array([0.0, 0.0, grasp_height])

    # 放置目标点
    above_goal = goal_position + np.array([0.0, 0.0, above_height])
    place_position = goal_position + np.array([0.0, 0.0, place_height])

    print("above_object:", above_object)
    print("grasp_position:", grasp_position)
    print("above_goal:", above_goal)
    print("place_position:", place_position)

    print("\nStep 1: open gripper")
    observation = hold_gripper(env, observation, gripper_action=1.0, steps=80)

    print("\nStep 2: move above object")
    observation = move_to_position(
        env,
        observation,
        above_object,
        gripper_action=1.0,
        max_steps=350,
        tolerance=0.015,
        speed_gain=3.0,
    )

    print("\nStep 3: move down to object")
    observation = move_to_position(
        env,
        observation,
        grasp_position,
        gripper_action=1.0,
        max_steps=350,
        tolerance=0.01,
        speed_gain=3.0,
    )

    print("\nStep 4: close gripper")
    observation = hold_gripper(env, observation, gripper_action=-1.0, steps=140)

    print("\nStep 5: lift object")
    observation = move_to_position(
        env,
        observation,
        above_object,
        gripper_action=-1.0,
        max_steps=350,
        tolerance=0.015,
        speed_gain=2.5,
    )

    print("\nStep 6: move above goal")
    observation = move_to_position(
        env,
        observation,
        above_goal,
        gripper_action=-1.0,
        max_steps=550,
        tolerance=0.015,
        speed_gain=2.2,
    )

    print("\nStep 7: move down to goal")
    observation = move_to_position(
        env,
        observation,
        place_position,
        gripper_action=-1.0,
        max_steps=350,
        tolerance=0.01,
        speed_gain=2.5,
    )

    print("\nStep 8: open gripper")
    observation = hold_gripper(env, observation, gripper_action=1.0, steps=140)

    print("\nStep 9: move up")
    observation = move_to_position(
        env,
        observation,
        above_goal,
        gripper_action=1.0,
        max_steps=350,
        tolerance=0.015,
        speed_gain=3.0,
    )

    final_object_position = observation["achieved_goal"].copy()
    final_goal_position = observation["desired_goal"].copy()
    final_error = np.linalg.norm(final_object_position - final_goal_position)

    print("\nRound finished.")
    print("final object position:", final_object_position)
    print("final goal position:", final_goal_position)
    print(f"final object-goal error: {final_error:.4f}")

    # 每轮结束后停一下，让你观察结果
    for _ in range(100):
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)


def main():
    env = gym.make(
        "PandaPickAndPlace-v3",
        render_mode="human",
        max_episode_steps=100000,
    )

    print("Looped Scripted Pick and Place Demo")
    print("每一轮都会自动生成新的方块位置和新的桌面目标点。")
    print("按 Ctrl+C 或关闭 PyBullet 窗口停止。")

    round_id = 1

    try:
        while True:
            run_one_pick_place_round(env, round_id)
            round_id += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()