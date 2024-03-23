# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import collections

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # boundary of action space
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) # use softplus to ensure std is positive
        dist = Normal(mu, std)
        normal_sample = dist.rsample()   # reparameterization trick
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)  # 1e-7 for numerical stability
        action = action * self.action_bound
        return action, log_prob


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

class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, 
                 actor_lr, critic_lr, alpha_lr, target_entropy, 
                 tau, gamma, device):
        # initialize networks
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, 
                               action_bound).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # for entropy coefficient automatic adjustment
        # Use log alpha to enhance stability
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.tau = tau               # soft update coefficient
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]
    
    def calc_target_Q(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        target_Q1 = self.target_critic_1(next_states, next_actions)
        target_Q2 = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(target_Q1, target_Q2) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def update(self, transition_dict):
        # extract data from transition_dict
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.FloatTensor(transition_dict['actions']
                                    ).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']
                                    ).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']
                                  ).view(-1, 1).to(self.device)
        # update critic networks
        td_target = self.calc_target_Q(rewards, next_states, dones)
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), 
                                   td_target.detach())  # td_target is not related to critic's params
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target.detach())
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        # update actor
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1 = self.critic_1(states, new_actions)
        q2 = self.critic_2(states, new_actions)
        actor_loss = -(torch.min(q1, q2) + self.log_alpha.exp() * entropy).mean() # entropy:(n, action_dim), but q1/q2:(n, 1)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update alpha
        alpha_loss = torch.mean(self.log_alpha.exp() * 
                                (entropy - self.target_entropy).detach())
                                # detach from graph, entropy is constant, not related to alpha
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        # soft update target networks
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

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