# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC4%E7%AB%A0-%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95.ipynb)

import random
from itertools import accumulate

class ValueIteration:
    """ Value Iteration """
    def __init__(self, env, theta, gamma):
        self.env = env
        if hasattr(self.env, 'ncol') and hasattr(self.env, 'nrow'): 
            self.num_obs = self.env.ncol * self.env.nrow
            if isinstance(self.env.action_space, list): # for self-defined cliff walking env
                self.num_actions = len(self.env.action_space)
            else: # frozen lake env
                self.num_actions = self.env.action_space.n
        else: # openai gym env
            self.num_obs = self.env.observation_space.n
            self.num_actions = self.env.action_space.n
        self.v = [0] * self.num_obs  # init state value
        self.theta = theta  # convergence threshold for state value
        self.gamma = gamma
        # policy
        self.pi = [None for i in range(self.num_obs)]

    def value_iteration(self):
        cnt = 1  # iteration count
        while 1:
            max_diff = 0
            new_v = [0] * self.num_obs
            for s in range(self.num_obs):
                qsa_list = []  # calculate all Q(s,a) values for state s
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] *
                                    (1 - done))
                    qsa_list.append(qsa)  # This line and next line are the main difference between value iteration and policy iteration.
                new_v[s] = max(qsa_list)  # value iteration uses bellman optimality equation
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # fulfill convergence condition, break out of loop
            cnt += 1
        print("Value iteration converged in %d iterations." % cnt)
        self.get_policy()

    def get_policy(self):  # get a greedy policy based on the current value function
        for s in range(self.num_obs):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # quantity of actions with maximum Q-value
            # assign the probability sum of 1 to actions with max Q, equally
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
    
    def act(self, state):
        '''
        choose an action based on the current policy
        
        首先, 对当前状态下的动作概率进行累加, 得到一个列表action_probs, 列表的元素为累加和.
        然后, 随机生成一个0到1之间的随机数, 根据随机数是否落于[action_probs[i-1], action_probs[i]]之间(若i=0, 则为[0, action_probs[0]]), 来决定选择哪个动作.
        
        例如原本的动作概率为[0.1, 0.2, 0.3, 0.4], 则累加和action_probs为[0.1, 0.3, 0.6, 1.0].
        若随机数为0.1, 在[0, 0.1], 选择动作0; 若随机数为0.2, 在[0.1, 0.3], 选择动作1, 依此类推.
        
        基本思想是利用均匀分布, 将概率转化为区间.
        '''
        action_probs = list(accumulate(self.pi[state]))
        random_num = random.random()
        for i, prob in enumerate(action_probs):
            if i == 0 and random_num < prob:
                return i
            elif random_num < prob and random_num >= action_probs[i-1]:
                return i