import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from alg.a2c import A2CDiscrete
from runner.on_policy_runner import OnPolicyRunner

# agent hyperparameters
actor_lr = 1e-3
critic_lr = 1e-2
hidden_dim = 128
gamma = 0.98
lamda = 0
entropy_coefficient = 0.01
# training hyperparameters
num_episodes = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environment hyperparameters
env_name = "CartPole-v1"

def render_result(agent):
    env = gym.make(env_name, render_mode="human")
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
    env.close()

def main():
    # create environment
    env = gym.make(env_name)
    # set seeds for reproducibility
    torch.manual_seed(0)
    # create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent1 = A2CDiscrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lamda, 0.01, device)
    agent2 = A2CDiscrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lamda, 1, device)
    # create runner
    runner = OnPolicyRunner(env, agent1, num_episodes)
    return_list = runner.run()
    runner.set_agent(agent2)
    return_list2 = runner.run()
    # plot return curve
    episode_list = np.arange(len(return_list))
    plt.plot(episode_list, return_list, label="entropy_coe = 0.01")
    plt.plot(episode_list, return_list2, label="entropy_coe = 1")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Actor-Critic on {}".format(env_name))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
