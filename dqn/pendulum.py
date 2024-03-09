import gymnasium as gym
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from alg.dueling_dqn import DQN, ReplayBuffer

# agent parameters
lr = 1e-2 # alpha in q-learning is replaced by lr in DQN
action_dim = 11 # Discretization of action space. Divide action space(-2,2) into 11 bins.
hidden_dim = 128  # number of neurons in the hidden layer
gamma = 0.98  # discount factor
epsilon = 0.01  # exploration rate
target_update = 50  # number of steps between target network updates
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use GPU if available
dqn_type = 'DuelingDQN' # DQN type, 'VanillaDQN' or 'DoubleDQN' or 'DuelingDQN'
# replay buffer parameters
buffer_size = 5000  # size of replay buffer
minimal_size = 1000  # minimum number of experience to start updating agent
batch_size = 64 # size of batch for training
# environment parameters
env_name = 'Pendulum-v1'  # name of the environment to train on
# training parameters
num_pbar = 10 # number of progress bars to display
num_episodes = 200  # number of episodes to train for


def set_seed(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed)

def dis_to_con(discrete_action, env, action_dim):
    """
    Convert discrete action to continuous action.
    """
    action_lowbound = env.action_space.low[0] # get the first element from the list
    action_upbound = env.action_space.high[0]
    action_con = action_lowbound + (action_upbound - action_lowbound) * discrete_action / (action_dim - 1)
    return np.array([action_con])


def train_dqn(agent, env, num_episodes, replay_buffer, minimal_size, batch_size, num_pbar):
    """
    Train DQN agent on environment.
    """
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(num_pbar):
        with tqdm(total=int(num_episodes/num_pbar), desc='Iteration %d'%i) as pbar:
            for i_episode in range(int(num_episodes/num_pbar)):
                episode_return = 0
                obs, info = env.reset()
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    # interact with environment
                    # 先得到action或先得到max_q_value都一样，因为这个过程中q_net不更新
                    action = agent.take_action(obs.reshape(1, -1))
                    max_q_value = agent.max_q_value(obs.reshape(1, -1))*0.005 + max_q_value*0.995
                    max_q_value_list.append(max_q_value) # 保存每个状态的最大Q值
                    action_con = dis_to_con(action, env, action_dim) # convert discrete action to continuous action
                    # print('action:', action, 'action_con:', action_con)
                    next_obs, rew, terminated, truncated, _ = env.step(action_con)
                    # add experience to replay buffer
                    replay_buffer.add(obs, action, rew, next_obs, terminated)
                    # update obs and episode return
                    obs = next_obs
                    episode_return += rew
                    # train agent if enough experience in replay buffer
                    if replay_buffer.size() > minimal_size:
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
    return return_list, max_q_value_list

def main(argv):
    env = gym.make(env_name)
    set_seed(env, 0) # set seed for reproducibility
    replay_buffer = ReplayBuffer(buffer_size) # create replay buffer
    obs_dim = env.observation_space.shape[0]  # observation dimension
    agent = DQN(obs_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type) # create agent
    return_list, max_q_value_list = train_dqn(agent, env, num_episodes, replay_buffer, minimal_size, batch_size, num_pbar) # train agent
    
    # show the trained agent in env
    env = gym.make(env_name, render_mode='human')
    obs, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = agent.take_action(obs.reshape(1, -1))
        action_con = dis_to_con(action, env, action_dim) # convert discrete action to continuous action
        next_obs, rew, terminated, truncated, _ = env.step(action_con)
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

    # plot max q value curve
    frame_list = range(len(max_q_value_list))
    plt.plot(frame_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--') # draw a orange horizontal line at y=0, line style is '--'
    plt.axhline(10, c='red', ls='--') # draw a red horizontal line at y=10, line style is '--'
    plt.xlabel('Frames')
    plt.ylabel('Max Q Values')
    plt.title('DQN on {}'.format(env_name))
    plt.show()            

if __name__ == '__main__':
    main(sys.argv[1:])
    # pass