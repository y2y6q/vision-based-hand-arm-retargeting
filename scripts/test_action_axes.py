import time

import gymnasium as gym
import numpy as np
import panda_gym


def run_action(env, action, name, steps=120):
    print(f"\nTesting: {name}")
    print("action =", action)

    observation, info = env.reset()

    time.sleep(0.8)

    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.01)

        if terminated or truncated:
            break

    time.sleep(0.8)


def main():
    env = gym.make("PandaPickAndPlace-v3", render_mode="human")

    observation, info = env.reset()

    print("PandaPickAndPlace action axes test")
    print("Action space:", env.action_space)
    print("Observation keys:", observation.keys())

    test_actions = [
        (np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "+X"),
        (np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), "-X"),

        (np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), "+Y"),
        (np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32), "-Y"),

        (np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), "+Z"),
        (np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32), "-Z"),

        (np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), "gripper +"),
        (np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32), "gripper -"),
    ]

    for action, name in test_actions:
        run_action(env, action, name)

    env.close()


if __name__ == "__main__":
    main()
