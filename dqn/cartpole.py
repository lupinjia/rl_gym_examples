import gymnasium as gym
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from alg.dqn import DQN, ReplayBuffer


lr = 2e-3 # alpha in q-learning is replaced by lr in DQN
num_episodes = 250  # number of episodes to train for
hidden_dim = 128  # number of neurons in the hidden layer
gamma = 0.98  # discount factor
epsilon = 0.01  # exploration rate
target_update = 10  # number of steps between target network updates
buffer_size = 10000  # size of replay buffer
batch_size = 64 # size of batch for training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use GPU if available
env_name = 'CartPole-v1'  # name of the environment to train on
num_pbar = 10 # number of progress bars to display

def set_seed(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)

def main(argv):
    env = gym.make(env_name)
    set_seed(env, 0) # set seed for reproducibility
    replay_buffer = ReplayBuffer(buffer_size) # create replay buffer
    obs_dim = env.observation_space.shape[0]  # observation dimension
    action_dim = env.action_space.n  # action dimension
    agent = DQN(obs_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device) # create agent
    return_list = []
    for i in range(num_pbar):
        with tqdm(total=int(num_episodes/num_pbar), desc='Iteration %d'%i) as pbar:
            for i_episode in range(int(num_episodes/num_pbar)):
                episode_return = 0
                obs, info = env.reset()
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    # interact with environment
                    action = agent.take_action(obs.reshape(1, -1))
                    next_obs, rew, terminated, truncated, _ = env.step(action)
                    # add experience to replay buffer
                    replay_buffer.add(obs, action, rew, next_obs, terminated)
                    # update obs and episode return
                    obs = next_obs
                    episode_return += rew
                    # train agent if enough experience in replay buffer
                    if replay_buffer.size() > batch_size:
                        obs_batch, action_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': obs_batch,
                            'actions': action_batch,
                            'rewards': rew_batch,
                            'next_states': next_obs_batch,
                            'dones': done_batch
                        }
                        agent.update(transition_dict)
                # add episode return to return list, for reward curve
                return_list.append(episode_return)
                # print episode number and mean return every 10 episodes
                if i_episode % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes/num_pbar*i + i_episode + 1),
                        'return':
                        '%.3f' % (np.mean(return_list[-10:]))
                    })
                # update progress bar
                pbar.update(1)
    # show the trained agent in env
    env = gym.make(env_name, render_mode='human')
    obs, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = agent.take_action(obs.reshape(1, -1))
        next_obs, rew, terminated, truncated, _ = env.step(action)
        env.render()
        obs = next_obs    
    env.close()

    # plot reward curve
    episode_list = range(len(return_list))
    plt.plot(episode_list, return_list)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DQN on {}'.format(env_name))
    plt.show()            

if __name__ == '__main__':
    main(sys.argv[1:])