import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from alg.pg import PolicyGradientDiscrete
from runner.on_policy_runner import OnPolicyRunner

# agent params
learning_rate = 1e-3
gamma = 0.98
hidden_dim = 128
# training params
num_episodes = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environment params
env_name = "CartPole-v1"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def render_result(env_name, agent: PolicyGradientDiscrete):
    # create environment
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
    env.close()

def main():
    # create environment
    env = gym.make(env_name)
    set_seed(0)
    # create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PolicyGradientDiscrete(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
    # create runner
    runner = OnPolicyRunner(env, agent, num_episodes)
    # train agent
    return_list = runner.run()
    # render agent
    render_result(env_name, agent)
    # plot episode return
    plt.plot(np.arange(len(return_list)), return_list)
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title("REINFORCE on {}".format(env_name))
    plt.show()

if __name__ == '__main__':
    main()
