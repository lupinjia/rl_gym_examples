# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import copy
import random
from itertools import accumulate

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        if hasattr(self.env, 'ncol') and hasattr(self.env, 'nrow'): 
            self.num_obs = self.env.ncol * self.env.nrow  # 状态数
            if isinstance(self.env.action_space, list): # self-defined cliff walking env
                self.num_actions = len(self.env.action_space)
            else: # frozen lake env
                self.num_actions = self.env.action_space.n
        else: # openai gym env
            self.num_obs = self.env.observation_space.n  # 状态数
            self.num_actions = self.env.action_space.n
        self.v = [0] * self.num_obs  # 价值函数初始化为0
        self.pi = [[1 / self.num_actions] * self.num_actions
                       for i in range(self.num_obs)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):
        """
        Policy Evaluation 
        策略评估 
        
        作用: 通过若干轮更新, 得到当前策略下, 真实价值函数的估计值.
        
        估计真实价值函数使用的是贝尔曼期望方程
        """
        cnt = 1  # 计数器
        while 1:
            max_diff = 0
            new_v = [0] * self.num_obs  # 新价值函数
            for s in range(self.num_obs):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] *
                                    (1 - done))
                        # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
                        # 本质上是要用奖励的期望: 对于确定值的奖励来说, 可以直接用奖励值, 相当于只有一种概率为1的可能; 
                                            # 但对于随机值奖励来说, 就需要像这样用状态转移概率乘以奖励来得到当前(s,a)下的奖励期望
                        # 同时,done=1表示终止状态,此时不能再进行转移,没有下一个状态,所以需要乘以(1-done)
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s])) # max_diff为标量,表示所有状态下价值函数的最大变化值
            self.v = new_v
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("Policy Evaluation Complete, Iteration: %d" % cnt)

    def policy_improvement(self):
        '''
        Policy Improvement
        策略提升
        
        作用: 利用策略评估得到的状态价值函数估计值, 改进当前策略, 使其更优.
        
        优化策略的方法是直接让动作价值函数最大的动作均分1的整体概率.
        '''
        for s in range(self.num_obs):
            # 每次都要重新计算所有状态的动作价值函数.
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] *
                                (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list) # 最大动作价值
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list] # 感觉比较粗糙
        print("Policy Improvement Complete")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,创建一个地址不同的新变量,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break # 达到最优策略
    
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