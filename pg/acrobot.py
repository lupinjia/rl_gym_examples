import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from alg.reinforce import REINFORCE

# agent params
learning_rate = 1e-3
gamma = 0.98
hidden_dim = 128
# training params
num_pbar = 10
num_episodes = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environment params
env_name = "Acrobot-v1"
action_type = "discrete"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    # create environment
    env = gym.make(env_name)
    set_seed(0)
    # create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device, action_type)
    # record episode return, for plotting
    return_list = []
    # train agent
    for i in range(num_pbar):
        with tqdm(total=int(num_episodes/num_pbar), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/num_pbar)):
                episode_return = 0 # return of the whole episode
                transition_dict = { # store transitions of the episode
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": []
                }
                state, _ = env.reset() # reset environment
                terminated, truncated = False, False
                while not terminated and not truncated:
                    action = agent.take_action(state) # select an action
                    next_state, reward, terminated, truncated, _ = env.step(action) # take action and get next state, reward, termination signal
                    # store transition
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(terminated)
                    state = next_state
                    episode_return += reward
                # add episode return to return list
                return_list.append(episode_return)
                # update agent per episode(MC)
                agent.update(transition_dict)
                if (i_episode+1)%10 == 0: # record agent performance every 10 episodes
                    pbar.set_postfix({
                        'episode_return':
                        '%d' % (num_episodes/num_pbar*i+i_episode+1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    render_result(env_name, agent)
    # plot episode return
    plt.plot(np.arange(len(return_list)), return_list)
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.title("REINFORCE on {}".format(env_name))
    plt.show()

def render_result(env_name, agent):
    # create environment
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
    env.close()


if __name__ == "__main__":
    main()