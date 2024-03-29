import numpy as np

class NStepSarsa:
    def __init__(self, env, epsilon, alpha, gamma, n):
        self.env = env
        try:
            self.num_obs = self.env.observation_space.n
        except:  # for blackjack env
            obs_space = self.env.observation_space 
            self.num_obs = 1
            for obs in obs_space:
                self.num_obs *= obs.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_obs, self.num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n  # n-step sarsa
        self.state_list = []   # store n-step experience
        self.action_list = []  
        self.reward_list = [] 
    
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
        if len(self.state_list) == self.n:  # ready for n-step update
            G = self.q_table[s1, a1]  # get Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]  # end to start, calculate G
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                # done=True时, 下一个状态为终止状态, 不会再次进入update函数.
                # 如果不在这里额外判断并进行更新的话, 终止状态前的n-1个状态的Q值将不会更新.
                # If reach the terminal state, the last few steps may not be enough to update n-step sarsa,
                # but we still need to update them to get the correct return.
                # If we don't add this extra check, the Q values of the last n-1 states will not be updated.
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.q_table[s, a] += self.alpha * (G - self.q_table[s, a])
            s = self.state_list.pop(0)  # pop the oldest state and action, which needs to be updated
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # main update procedure of n-step sarsa
            self.q_table[s, a] += self.alpha * (G - self.q_table[s, a])
        if done:  # If episode ends, reset state_list, action_list, reward_list, prepare for next episode
            self.state_list = []
            self.action_list = []
            self.reward_list = []