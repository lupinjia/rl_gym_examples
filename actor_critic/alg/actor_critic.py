# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # convert to probabilities of actions, using softmax

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # ensure std is positive
        return mu, std

class ValueNet(nn.Module): 
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # 状态s下对应的值函数为标量, 所以输出维度固定为1
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x) # 输出值函数

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device, action_type):
        if action_type == 'discrete':
            self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # hyperparameters
        self.action_type = action_type
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        if self.action_type == 'discrete':
            probs = self.actor(state)  # get probabilities of actions
            action_dist = torch.distributions.Categorical(probs)  # create categorical distribution
            action = action_dist.sample()  # sample action from distribution
            return action.item()
        else:
            mu, std = self.actor(state)  # get mean and std of actions
            action_dist = torch.distributions.Normal(mu, std)  # create normal distribution
            action = action_dist.sample()  # sample action from distribution
            return [action.item()]
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)  # calc temporal difference error
        if self.action_type == 'discrete':
            log_probs = torch.log(self.actor(states).gather(1, actions))
        else:
            mu, std = self.actor(states)
            log_probs = torch.distributions.Normal(mu, std).log_prob(actions)
        actor_loss = torch.mean(-log_probs * td_error.detach())  # detach()防止梯度传递, 共享数据内存, 值相同
                                                                 # td_error is not related to actor parameter, can be seen as constant
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())
        # prepare to update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # calc gradients
        actor_loss.backward()
        critic_loss.backward()
        # update weights
        self.actor_optimizer.step()
        self.critic_optimizer.step()