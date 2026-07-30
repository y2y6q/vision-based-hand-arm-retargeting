import time

import gymnasium as gym
import numpy as np

import panda_gym


def main():
    env = gym.make(
        "PandaAllegroPickAndPlace-v0",
        render_mode="human",
    )

    observation, info = env.reset()

    print("=" * 80)
    print("PandaAllegroPickAndPlace-v0 loaded successfully.")
    print("=" * 80)
    print("action_space:", env.action_space)
    print("observation keys:", observation.keys())

    for key, value in observation.items():
        print(f"{key}: shape={np.array(value).shape}")

    print("\nStart random action test.")

    for step in range(1000):
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        # 前 300 步：机械臂轻微移动
        if step < 300:
            action[0] = 0.2

        # 300-600 步：Allegro 手指弯曲
        elif step < 600:
            action[3:] = 0.5

        # 600-900 步：Allegro 手指张开
        elif step < 900:
            action[3:] = -0.5

        observation, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            print(f"step={step}, reward={reward}, terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(1.0 / 240.0)

    print("\nTest finished.")
    print("Close PyBullet window or press Ctrl+C to stop.")

    try:
        while True:
            env.render()
            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()