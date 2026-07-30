import time

import gymnasium as gym
import numpy as np
import panda_gym


def main():
    env = gym.make("PandaReach-v3", render_mode="human")
    observation, info = env.reset()

    print("PandaReach controller demo")
    print("Action space:", env.action_space)
    print("Observation keys:", observation.keys())

    for step in range(1000):
        achieved_goal = observation["achieved_goal"]
        desired_goal = observation["desired_goal"]

        error = desired_goal - achieved_goal

        # PandaReach-v3 的 action 只有 3 维：[dx, dy, dz]
        action = 5.0 * error
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        observation, reward, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            distance = np.linalg.norm(error)
            print(f"step={step}, distance={distance:.4f}, action={action}")

        time.sleep(0.01)

        if terminated or truncated:
            print("reset environment")
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()