import numpy as np

class NStepSarsa:
    def __init__(self, env, epsilon, alpha, gamma, n):
        self.env = env
        self.num_obs = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_obs, self.num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n # n-step sarsa
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励
    
    def take_action(self, state): # epsilon-greedy policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[state])
        return action
    
    def update(self, s0, a0, r, s1, a1, done):
        # using list to store n-step experience
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
            G = self.q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]  # 不断向前计算每一步的回报
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                # done=True时, 下一个状态为终止状态, 不会再次进入update函数.
                # 如果不在这里额外判断并进行更新的话, 终止状态前的n-1个状态的Q值将不会更新.
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.q_table[s, a] += self.alpha * (G - self.q_table[s, a])
            s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n步Sarsa的主要更新步骤
            self.q_table[s, a] += self.alpha * (G - self.q_table[s, a])
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []