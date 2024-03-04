import numpy as np
import random

class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self,
                 env,
                 epsilon,
                 alpha,
                 gamma,
                 n_planning):
        self.env = env
        try:
            self.num_obs = self.env.observation_space.n
        except: # for blackjack env
            obs_space = self.env.observation_space 
            self.num_obs = 1
            for obs in obs_space:
                self.num_obs *= obs.n
        self.num_action = self.env.action_space.n  # 动作空间的维度
        self.q_table = np.zeros([self.num_obs, self.num_action])  # 初始化Q(s,a)表格
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def take_action(self, obs):  # 选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(self.q_table[obs])
        return action

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.q_table[s1].max(
        ) - self.q_table[s0, a0]
        self.q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)