import gymnasium as gym
import panda_gym


def main():
    env = gym.make("PandaPickAndPlace-v3", render_mode="human")

    observation, info = env.reset()

    print("PandaPickAndPlace 环境创建成功")
    print("Action space:", env.action_space)
    print("Observation keys:", observation.keys())
    print("Observation:", observation)

    for step in range(1500):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()