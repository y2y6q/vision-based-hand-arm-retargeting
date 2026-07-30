import time

import gymnasium as gym
import numpy as np
import panda_gym
import pybullet as p


ESC_KEY = 27


def get_keyboard_events(env):
    try:
        client = env.unwrapped.sim.physics_client
        return client.getKeyboardEvents()
    except Exception:
        return p.getKeyboardEvents()


def key_down(keys, key):
    return key in keys and keys[key] & p.KEY_IS_DOWN


def key_down_letter(keys, letter):
    """
    同时检测小写和大写，避免 W/w 不识别的问题。
    """
    return key_down(keys, ord(letter.lower())) or key_down(keys, ord(letter.upper()))


def main():
    env = gym.make(
        "PandaPickAndPlace-v3",
        render_mode="human",
        max_episode_steps=100000,
    )
    observation, info = env.reset()

    print("Keyboard control demo started")
    print("Action space:", env.action_space)
    print("")
    print("控制说明：")
    print("I / K : +X / -X")
    print("L / J : +Y / -Y")
    print("U / M : +Z / -Z")
    print("O / P : open / close gripper")
    print("R     : reset")
    print("ESC   : exit")
    print("")
    print("注意：运行后先用鼠标点一下 PyBullet 窗口，再按键。")

    action_scale = 0.7

    while True:
        keys = get_keyboard_events(env)

        try:
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        except Exception:
            pass

        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # X 方向
        if key_down_letter(keys, "i"):
            action[0] += action_scale
        if key_down_letter(keys, "k"):
            action[0] -= action_scale

        # Y 方向
        if key_down_letter(keys, "l"):
            action[1] += action_scale
        if key_down_letter(keys, "j"):
            action[1] -= action_scale

        # Z 方向
        if key_down_letter(keys, "u"):
            action[2] += action_scale
        if key_down_letter(keys, "m"):
            action[2] -= action_scale

        # gripper
        if key_down_letter(keys, "o"):
            action[3] += 1.0  # 张开
        if key_down_letter(keys, "p"):
            action[3] -= 1.0  # 闭合

        # reset
        if key_down_letter(keys, "r"):
            observation, info = env.reset()
            print("reset environment")
            time.sleep(0.3)
            continue

        # exit
        if key_down(keys, ESC_KEY):
            print("exit")
            break

        action = np.clip(action, -1.0, 1.0)

        observation, reward, terminated, truncated, info = env.step(action)



        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()