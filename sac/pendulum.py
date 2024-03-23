# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import gymnasium as gym
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from alg.sac import SAC, ReplayBuffer

# agent hyperparameters
actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005 # for soft update of target parameters
hidden_dim = 128
# replay buffer hyperparameters
buffer_size = 100000
minimal_size = 1000
batch_size = 64
# training hyperparameters
num_episodes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environment hyperparameters
env_name = "Pendulum-v1"

def train_off_policy_agent(env, agent, replay_buffer, num_episodes, minimal_size, batch_size):
    # to record episode returns
    return_list = []
    for i in range(10): # 10 pbars by default
        with tqdm(total=int(num_episodes/10), desc="Iteration %d"%(i)) as pbar:
            for i_episode in range(int(num_episodes/10)): # for each pbar, there are int(num_episodes/10) episodes
                # each episode
                episode_return = 0
                # reset environment
                state, _ = env.reset()
                terminated, truncated = False, False
                while not terminated and not truncated: # interaction loop
                    # select action
                    action = agent.take_action(state)
                    # step the environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    # store transition
                    replay_buffer.add(state, action, reward, next_state, terminated)
                    # update state
                    state = next_state
                    episode_return += reward
                    # update agent if enough transitions are in buffer
                    if replay_buffer.size() >= minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size) # batch states, actions, rewards, next_states, dones
                        transition_dict = {"states": b_s, "actions": b_a, "rewards": b_r, "next_states": b_ns, "dones": b_d}
                        agent.update(transition_dict)
                # episode finished
                return_list.append(episode_return)
                # show mean return of last 10 episodes, every 10 episodes
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        "episode":
                        "%d" % (int(num_episodes/10)*i+i_episode+1),
                        "return":
                        "%.3f" % (np.mean(return_list[-10:]))
                    })
                # update pbar
                pbar.update(1)
    # return return_list
    return return_list

def test_train_off_policy(env, agent, replay_buffer, minimal_size, batch_size):
    # to record episode returns
    return_list = []
    with tqdm(total=10, desc="Test 10 episodes") as pbar:
        for i_episode in range(10): # 10 episodes
            # each episode
            episode_return = 0
            # reset environment
            state, _ = env.reset()
            terminated, truncated = False, False
            while not terminated and not truncated: # interaction loop
                action = agent.take_action(state)
                # step the environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                # store transition
                replay_buffer.add(state, action, reward, next_state, terminated)
                # update state
                state = next_state
                # update agent if enough transitions are in buffer
                if replay_buffer.size() >= minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size) # batch states, actions, rewards, next_states, dones
                    transition_dict = {"states": b_s, "actions": b_a, "next_states": b_ns, "rewards": b_r, "dones": b_d}
                    agent.update(transition_dict)
                episode_return += reward
            # episode finished
            return_list.append(episode_return)
            # update pbar
            pbar.update(1)


def render_agent(env_name, agent):
    # reset environment
    env = gym.make(env_name, render_mode="human")
    state, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated: # interaction loop
        # select action
        action = agent.take_action(state)
        # step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        # render environment
        env.render()
        # update state
        state = next_state

def main():
    # create environment
    env = gym.make(env_name)
    # set seeds
    torch.manual_seed(0)
    random.seed(0)
    # create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)
    # create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]    # upper bound of action range
    target_entropy = -action_dim
    agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    ## test off-policy agent
    # test_train_off_policy(env, agent, replay_buffer, minimal_size, batch_size)
    # train agent
    return_list = train_off_policy_agent(env, agent, replay_buffer, num_episodes, minimal_size, batch_size)
    # plot return curve
    episodes_list = np.arange(len(return_list))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("SAC on {}".format(env_name))
    plt.show()
    # render agent
    render_agent(env_name, agent)

if __name__ == "__main__":
    main()
