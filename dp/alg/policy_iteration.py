# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import copy
import random
from itertools import accumulate

class PolicyIteration:
    """ Policy Iteration """
    def __init__(self, env, theta, gamma):
        self.env = env
        if hasattr(self.env, 'ncol') and hasattr(self.env, 'nrow'): 
            self.num_obs = self.env.ncol * self.env.nrow  # number of states
            if isinstance(self.env.action_space, list):  # self-defined cliff walking env
                self.num_actions = len(self.env.action_space)
            else:  # frozen lake env
                self.num_actions = self.env.action_space.n
        else:  # openai gym env
            self.num_obs = self.env.observation_space.n   # number of states
            self.num_actions = self.env.action_space.n
        self.v = [0] * self.num_obs   # value function init to 0
        self.pi = [[1 / self.num_actions] * self.num_actions
                       for i in range(self.num_obs)]   # init as a uniform stochastic policy
        self.theta = theta   # Policy Iteration convergence threshold
        self.gamma = gamma   # discount factor

    def policy_evaluation(self):
        """
        Policy Evaluation 

        estimate the value function of the current policy,
        using the bellman expectation equation.
        """
        cnt = 1  # counter
        while 1:
            max_diff = 0
            new_v = [0] * self.num_obs  # new value function
            for s in range(self.num_obs):
                qsa_list = []  # calcualte qsa for all actions of state s
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] *
                                    (1 - done))
                        # reward needs to be multiplied by the probability of transition,
                        # because the environment is stochastic.
                        # Essentially, we are using the expected value of reward
                        # When done is True, it is the terminator, next state's value is 0.
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # bellman expectation equation
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # fulfill the convergence threshold, exit the loop
            cnt += 1
        print("Policy Evaluation Complete, Iteration: %d" % cnt)

    def policy_improvement(self):
        '''
        Policy Improvement
        策略提升

        Make use of the estimated value function gained by policy evaluation, 
        to improve the current policy.

        The improvement approach is to 
        assign equal probability to the action with the highest action value.
        '''
        for s in range(self.num_obs):
            # calculate the action-value function for all actions of state s every time.
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] *
                                (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # how many actions have the max value
            # divide the probability to the action(s) with the highest action value, equally.
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]  # rough treatment
        print("Policy Improvement Complete")
        return self.pi

    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # create a new variable with different address, for comparison later.
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break  # reach the optimal policy
    
    def act(self, state):
        '''
        choose an action based on the current policy
        
        首先, 对当前状态下的动作概率进行累加, 得到一个列表action_probs, 列表的元素为累加和.
        然后, 随机生成一个0到1之间的随机数, 根据随机数是否落于[action_probs[i-1], action_probs[i]]之间(若i=0, 则为[0, action_probs[0]]), 来决定选择哪个动作.
        
        例如原本的动作概率为[0.1, 0.2, 0.3, 0.4], 则累加和action_probs为[0.1, 0.3, 0.6, 1.0].
        若随机数为0.1, 在[0, 0.1], 选择动作0; 若随机数为0.2, 在[0.1, 0.3], 选择动作1, 依此类推.
        
        基本思想是利用均匀分布, 将概率转化为区间.
        '''
        action_probs = list(accumulate(self.pi[state]))  # accumulate the action probabilities
        random_num = random.random()
        for i, prob in enumerate(action_probs):  # select the action based on the random number
            if i == 0 and random_num < prob:
                return i
            elif random_num < prob and random_num >= action_probs[i-1]:
                return i