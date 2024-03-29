import numpy as np
import random

class DynaQ:
    """ Dyna-Q Algorithm """
    def __init__(self,
                 env,
                 epsilon,
                 alpha,
                 gamma,
                 n_planning):
        self.env = env
        try:
            self.num_obs = self.env.observation_space.n
        except:  # for blackjack env
            obs_space = self.env.observation_space 
            self.num_obs = 1
            for obs in obs_space:
                self.num_obs *= obs.n
        self.num_action = self.env.action_space.n
        self.q_table = np.zeros([self.num_obs, self.num_action])  # init Q-table to zero
        self.alpha = alpha  # step size
        self.gamma = gamma
        self.epsilon = epsilon  # exploration

        self.n_planning = n_planning  # times of Q-planning for one Q-learning
        self.model = dict()  # env model

    def take_action(self, obs):
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
        self.model[(s0, a0)] = r, s1  # add data to model
        for _ in range(self.n_planning):  # Q-planning loop
            # Randomly select a state-action pair(which has been visited) from the model
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)