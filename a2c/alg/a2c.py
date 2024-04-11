# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        '''
        state_dim: int, input state dimension
        hidden_dim: list of hidden dimensions, e.g. [32, 16]
                    or int(1 hidden layer), e.g. 32
        action_dim: int, output action dimension
        '''
        super(PolicyNetDiscrete, self).__init__()
        #----- Hidden Dimension List Process -----#
        if isinstance(hidden_dim, int):  # if only one hidden layer
            hidden_dim = [hidden_dim]
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim[0]))  # hidden layer 1
        layers.append(nn.ReLU())  # activation function
        for i in range(len(hidden_dim)):  # 
            if i == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[i], action_dim))
            else:
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=1))  # discrete action, use softmax as output activation
        self.net = nn.Sequential(*layers)
        #----- Hidden Dimension List Process -----#
    
    def forward(self, x):
        return self.net(x)

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        '''
        state_dim: int, input state dimension
        hidden_dim: list of hidden dimensions, e.g. [32, 16]
                    or int(1 hidden layer), e.g. 32
        action_dim: int, output action dimension
        '''
        super(PolicyNetContinuous, self).__init__()
        #----- Hidden Dimension List Process -----#
        if isinstance(hidden_dim, int):  # enable single hidden layer
            hidden_dim = [hidden_dim]
        layers = []                      # generate layers according to input list, which contains the number of neurons in each layer
        layers.append(nn.Linear(state_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[i], action_dim))
            else:
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        #----- Hidden Dimension List Process -----#
        self.std = nn.Parameter(0.1 * torch.ones(action_dim))
    
    def forward(self, x):
        mean = self.net(x)
        return mean, self.std

class ValueNet(nn.Module): 
    def __init__(self, state_dim, hidden_dim):
        '''
        state_dim: int, input state dimension
        hidden_dim: list of hidden dimensions, e.g. [32, 16]
                    or int(1 hidden layer), e.g. 32
        '''
        super(ValueNet, self).__init__()
        #----- Hidden Dimension List Process -----#
        if isinstance(hidden_dim, int):  # enable single hidden layer
            hidden_dim = [hidden_dim]
        layers = []                      # generate layers according to input list, which contains the number of neurons in each layer
        layers.append(nn.Linear(state_dim, hidden_dim[0]))  # layer 1
        layers.append(nn.ReLU())  # activation function
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[i], 1))  # output layer
            else:
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        #----- Hidden Dimension List Process -----#
    
    def forward(self, x):
        return self.net(x)

class A2CDiscrete:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lamda, entropy_coefficient, device):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dist = torch.distributions.Categorical(torch.tensor([1/action_dim for _ in range(action_dim)]))
        # hyperparameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coe = entropy_coefficient
        self.device = device
    
    def update_dist(self, action_probs):
        # update action distribution based on action probabilities
        self.action_dist = torch.distributions.Categorical(action_probs)
    
    def select_action(self, state):
        # select action based on current policy
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        probs = self.actor(state)  # get probabilities of actions
        self.update_dist(probs)  # update action distribution
        action = self.action_dist.sample()  # sample action from distribution
        return action.item()  # discrete action, return int scalar
    
    def compute_returns_and_advantages(self, transition_dict, lamda):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).view(-1, self.state_dim).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).view(-1, self.state_dim).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # compute advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        values = self.critic(states)
        td_error = td_target - values  # calc temporal difference error
        advantages = self.compute_gae(td_error, dones, lamda)
        returns = advantages + values
        return returns, advantages
    
    def compute_gae(self, td_error, dones, lamda):
        gae = torch.zeros_like(td_error)  # gae's shape is the same as td_error(length: samples(transitions)*1)
        gae[-1] = td_error[-1]
        for t in reversed(range(len(td_error) - 1)):
            gae[t] = td_error[t] + (1 - dones[t]) * self.gamma * lamda * gae[t + 1]
        return gae
    
    def learn(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).view(-1, self.state_dim).to(self.device)
        # convert list of ndarrays to ndarray, cause converting list of ndarrays to tensor is extremely slow
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)

        returns, advantages = self.compute_returns_and_advantages(transition_dict, self.lamda)
        log_probs = torch.log(self.actor(states).gather(1, actions))  # elements in actions must be subscripts to use gather()
        actor_loss = torch.mean(-log_probs * advantages.detach())  # detach()防止梯度传递, 共享数据内存, 值相同
                                                                   # td_error is not related to actor parameter, can be seen as constant
                                                                   # also prevent the gradient accumulation in critic params
        #----- Adding Entropy Loss -----#
        self.update_dist(self.actor(states))  # update action distribution
        entropy = torch.mean(self.action_dist.entropy())
        entropy_loss = -self.entropy_coe * entropy
        actor_loss = actor_loss + entropy_loss
        #----- Adding Entropy Loss -----#
        critic_loss = F.mse_loss(self.critic(states), returns.detach())  # if not calculate the graident of critic target(returns) w.r.t. 
                                                                         # critic params, the performance will be slightly better(see curve/comparison_CriticTargetGradient.png)
        # update actor and critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()