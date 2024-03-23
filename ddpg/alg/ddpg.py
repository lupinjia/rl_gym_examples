# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # boundary of action space
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound  # tanh activation(output range [-1, 1]) 
                                                            # and scale action to [-action_bound, action_bound]

class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim) # input state and action
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1) # output scalar Q(s,a)
    
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # concatenate state and action
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x) # output scalar Q(s,a)

class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        # initialize networks
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # init params of target networks with same weights as original networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # hyperparameters
        self.action_dim = action_dim
        self.sigma = sigma           # standard deviation of exploration noise(gaussian)
        self.tau = tau               # soft update coefficient
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = self.actor(state).item() #?
        # add exploration noise(gaussian)
        action += self.sigma * np.random.randn(self.action_dim)
        return action
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def update(self, transition_dict):
        # extract data from transition_dict
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.FloatTensor(transition_dict['actions']).view(-1, self.action_dim).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        # print("shape of next_states:", next_states.shape)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        # calculate target Q value
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_target = rewards + self.gamma * (1 - dones) * next_q_values
        # update critic
        critic_loss = F.mse_loss(self.critic(states, actions), q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # soft update target networks
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size) # return a list of random transitions(tuples)
        states, actions, rewards, next_states, dones = zip(*transitions) # unpack the list
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def size(self):
        return len(self.buffer)