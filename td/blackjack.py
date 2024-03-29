import gymnasium as gym
import sys
from tqdm import tqdm  # tqdm is a progress bar library
import matplotlib.pyplot as plt
import numpy as np

from alg.sarsa import SARSA
from alg.q_learning import QLearning
from alg.n_step_sarsa import NStepSarsa
from alg.dyna_q import DynaQ
from utils.arg import parse_args

# np.random.seed(0)
# some hyperparameters
epsilon = 0.1 # epsilon越大, 算法越倾向于随机选择动作, 越有可能探索更多的状态空间
# The bigger the epsilon, the more random the agent will be.
alpha = 0.1 # learning rate alpha. 加大更新力度，让下一个状态的Q值对当前状态Q值的影响更大一些
gamma = 0.9
num_episodes = 1500 # number of episodes to run. 500个episode时可能学不出来太好的效果，加大episode数可以在一定范围内提升学习效果
num_pbar = 10 # number of progress bar
n_steps = 5 # n-steps SARSA
n_planning = 5 # number of planning steps for Dyna-Q

def main(argv):
    # blackjack env: https://gymnasium.farama.org/environments/toy_text/blackjack/
    env = gym.make('Blackjack-v1')
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
                obs = transform_blackjack_obs(obs) # transform blackjack observation(3 elements tuple) to a scalar
                action = agent.take_action(obs) # transform original obs to scalar, for Q table
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    # SARSA needs next_obs and next_action
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_obs = transform_blackjack_obs(next_obs)
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
    # evaluate the learned policy
    eval_blackjack_policy(env, agent)
    
    # plot the return curve
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SARSA on Cliff-Walking Env')
    plt.show()

def transform_blackjack_obs(obs):
    # transform blackjack observation(3 elements tuple) to a scalar
    return ((obs[0]*11 + obs[1]) * 2) + obs[2]

def eval_blackjack_policy(env, agent, eval_time=10000):
    # evaluate the learned policy for eval_time episodes
    win, lose, draw = 0, 0, 0
    for i in range(eval_time):
        obs, _ = env.reset()
        obs = transform_blackjack_obs(obs)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = agent.take_action(obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            obs = transform_blackjack_obs(obs)
            if terminated or truncated:
                if rew == 1:
                    win += 1
                elif rew == -1:
                    lose += 1
                else:
                    draw += 1
    print("Evaluation Result")
    print("Win: %d, Lose: %d, Draw: %d"%(win, lose, draw))
    print("Win Rate: %.3f, Lose Rate: %.3f, Draw Rate: %.3f"%(win/eval_time, lose/eval_time, draw/eval_time))

if __name__ == '__main__':
    main(sys.argv[1:])