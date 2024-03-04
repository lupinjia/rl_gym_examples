import gymnasium as gym
import sys
from tqdm import tqdm  # tqdm是显示循环进度条的库
import matplotlib.pyplot as plt
import numpy as np

from alg.sarsa import SARSA
from alg.q_learning import QLearning
from alg.n_step_sarsa import NStepSarsa
from alg.dyna_q import DynaQ
from utils.arg import parse_args

np.random.seed(10)
# some hyperparameters
epsilon = 0.1 # epsilon越大, 算法越倾向于随机选择动作, 越有可能探索更多的状态空间
alpha = 0.1 # learning rate alpha
gamma = 0.9
num_episodes = 500 # number of episodes to run
num_pbar = 10 # number of progress bar
n_steps = 5 # 5-steps SARSA
n_planning = 2 # number of planning steps for Dyna-Q


def main(argv):
    # cliff walking env: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make('CliffWalking-v0')
    is_q_learning, is_n_step, is_dyna_q = parse_args(argv)
    if is_q_learning: # if use Q-Learning
        agent = QLearning(env, epsilon, alpha, gamma)
    elif is_n_step: # if use n-step SARSA
        agent = NStepSarsa(env, epsilon, alpha, gamma, n_steps)
    elif is_dyna_q: # if use Dyna-Q
        agent = DynaQ(env, epsilon, alpha, gamma, n_planning)
    else:
        agent = SARSA(env, epsilon, alpha, gamma)

    return_list = []
    # run the agent for num_episodes episodes
    for i in range(num_pbar):
        with tqdm(total=int(num_episodes/num_pbar), desc='Iteration %d'%i) as pbar:
            for i_episode in range(int(num_episodes/num_pbar)): # each pbar has (num_episodes/num_pbar) episodes
                episode_return = 0 # initialize episode return
                obs, _ = env.reset() # reset env at episode beginning
                action = agent.take_action(obs)
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    # SARSA needs next_obs and next_action
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_action = agent.take_action(next_obs)
                    # episode return accumulated
                    episode_return += reward # do not consider gamma
                    if is_n_step: # n-step sarsa needs done flag
                        agent.update(obs, action, reward, next_obs, next_action, terminated or truncated)
                    elif is_dyna_q or is_q_learning:
                        agent.update(obs, action, reward, next_obs)
                    else:
                        agent.update(obs, action, reward, next_obs, next_action)
                    obs = next_obs
                    action = next_action
                # update return list
                return_list.append(episode_return)
                if (i_episode+1)%10 == 0: # print mean episode return every 10 episodes
                    pbar.set_postfix({
                        'episode':
                        '%d'%(int(num_episodes/num_pbar)*i+i_episode+1),
                        'return':
                        '%.3f'%(np.mean(return_list[-10:])) # show the mean of last 10 episodes return 
                    }) 
                # update progress bar
                pbar.update(1)
    # demonstrate the learned policy
    env = gym.make('CliffWalking-v0', render_mode='human')
    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = agent.take_action(obs)
        obs, rew, terminated, truncated, _ = env.step(action)
        # print('Truncated: ', truncated, 'Terminated: ', terminated)
        env.render()
    env.close()

    # print Q table
    # print("Q table: \n", agent.Q_table)
    
    # plot the return curve
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SARSA on Cliff-Walking Env')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])

