import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from alg.actor_critic import ActorCritic

# agent hyperparameters
actor_lr = 1e-3
critic_lr = 1e-2
hidden_dim = 128
gamma = 0.98
# training hyperparameters
num_episodes = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# environment hyperparameters
env_name = "CartPole-v0"

def train_on_policy_agent(env, agent, num_episodes):
    # to record episode returns
    return_list = []
    for i in range(10): # 10 pbars by default
        with tqdm(total=int(num_episodes/10), desc="Iteration %d"%(i)) as pbar:
            for i_episode in range(int(num_episodes/10)): # for each pbar, there are int(num_episodes/10) episodes
                # each episode
                episode_return = 0
                transition_dict = {
                    "states":[],
                    "actions":[],
                    "next_states":[],
                    "rewards":[],
                    "dones":[]
                }
                # reset environment
                state, _ = env.reset()
                terminated, truncated = False, False
                while not terminated and not truncated: # interaction loop
                    # select action
                    action = agent.take_action(state)
                    # step the environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    # store transition
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(terminated)
                    # update state
                    state = next_state
                    episode_return += reward
                # episode finished
                return_list.append(episode_return)
                # update agent
                agent.update(transition_dict)
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

def main():
    # create environment
    env = gym.make(env_name)
    # set seeds for reproducibility
    torch.manual_seed(0)
    # create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
    # train agent
    return_list = train_on_policy_agent(env, agent, num_episodes)
    # plot return curve
    episode_list = np.arange(len(return_list))
    plt.plot(episode_list, return_list)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Actor-Critic on {}".format(env_name))
    plt.show()

if __name__ == "__main__":
    main()
