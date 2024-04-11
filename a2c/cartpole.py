import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from alg.a2c import A2CDiscrete
from runner.on_policy_runner import OnPolicyRunner

# agent hyperparameters
actor_lr = 1e-3
critic_lr = 1e-2
hidden_dim = 128
gamma = 0.98
lamda = 0
# training hyperparameters
num_episodes = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environment hyperparameters
env_name = "CartPole-v1"

def main():
    # create environment
    env = gym.make(env_name)
    # set seeds for reproducibility
    torch.manual_seed(0)
    # create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent_lamda0 = A2CDiscrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, 0, device)
    agent_lamda1 = A2CDiscrete(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, 1, device)
    # create runner
    runner = OnPolicyRunner(env, agent_lamda0, num_episodes)
    return_list_lamda0 = runner.run()
    runner.set_agent(agent_lamda1)
    return_list_lamda1 = runner.run()
    # plot return curve
    episode_list = np.arange(len(return_list_lamda0))
    plt.plot(episode_list, return_list_lamda0, label="$\lambda=0$")
    plt.plot(episode_list, return_list_lamda1, label="$\lambda=1$")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Actor-Critic on {}".format(env_name))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
