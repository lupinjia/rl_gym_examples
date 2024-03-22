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
        x = F.softmax(self.fc2(x), dim=1) # output probabilities of finite discrete actions
        return x

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class PPODiscrete:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # initialize policy and value networks
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # hyperparameters
        self.lmbda = lmbda
        self.epochs = epochs # Number of epochs trained using each trajectory
                             # In PPO, training on a whole trajectory is called "epoch", 
                             # each "batch" has a fixed number of transitions,
                             # a transition consists of (state, action, reward, next_state, done)
        self.eps = eps       # Epsilon, specify the clipping range in PPO-clip
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action_probs = self.actor(state)
        action = torch.distributions.Categorical(action_probs).sample()
        return action.item()
    
    def update(self, transition_dict):
        # The transition_dict contains a list of transitions for a single episode.
        # Convert transition_dict to tensors
        states = torch.tensor(transition_dict['states'], 
                              dtype=torch.float).to(self.device) # shape: (episode_length, state_dim)
        actions = torch.tensor(transition_dict['actions'], 
                               dtype=torch.long).view(-1, 1).to(self.device) # shape: (episode_length, 1)
        rewards = torch.tensor(transition_dict['rewards'], 
                               dtype=torch.float).view(-1, 1).to(self.device) # shape: (episode_length, 1)
        next_states = torch.tensor(transition_dict['next_states'], 
                                   dtype=torch.float).to(self.device) # shape: (episode_length, state_dim
        dones = torch.tensor(transition_dict['dones'], 
                             dtype=torch.float).view(-1, 1).to(self.device) # shape: (episode_length, 1)
        # Compute advantages
        td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_targets - self.critic(states) # Temporal Difference Error(can also be called advantage in actor-critic)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device) # compute advantage using GAE
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions)) # target policy log probabilities
            ratio = torch.exp(log_probs - old_log_probs) # importance weight
            surr1 = ratio * advantage                                      # first term in PPO-clip loss, surrogate loss 1
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage # second term in PPO-clip loss, surrogate loss 2
            actor_loss = -torch.min(surr1, surr2).mean()  # PPO-clip loss
            critic_loss = F.mse_loss(self.critic(states), td_targets.detach()) # critic seeks to approximate td_targets
            # Update actor and critic networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def compute_advantage(self, gamma, lmbda, td_delta):
        'Using Generalized Advantage Estimation(GAE)'
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]: # reverse order
            advantage = delta + gamma * lmbda * advantage # Deltas that are more close to the epsode end will multiply more times gamma and lambda
            advantage_list.append(advantage)
        advantage_list.reverse() # reverse order back to original
        return torch.tensor(advantage_list, dtype=torch.float)

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim) # output mean of action distribution
        self.fc_std = nn.Linear(hidden_dim, action_dim) # output standard deviation of action distribution
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) # Use softplus to ensure std is always positive.
                                         # (https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
        return mu, std

class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # initialize policy and value networks
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # hyperparameters
        self.lmbda = lmbda
        self.epochs = epochs # Number of epochs trained using each trajectory
        self.eps = eps       # Epsilon, specify the clipping range in PPO-clip
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action = torch.distributions.Normal(mu, sigma).sample()
        return [action.item()]
    
    def update(self, transition_dict):
        # The transition_dict contains a list of transitions for a single episode.
        # Convert transition_dict to tensors
        states = torch.tensor(transition_dict['states'], 
                              dtype=torch.float).to(self.device) # shape: (episode_length, state_dim)
        actions = torch.tensor(transition_dict['actions'], 
                               dtype=torch.float).view(-1, 1).to(self.device) # shape: (episode_length, 1)
        rewards = torch.tensor(transition_dict['rewards'], 
                               dtype=torch.float).view(-1, 1).to(self.device) # shape: (episode_length, 1)
        next_states = torch.tensor(transition_dict['next_states'], 
                                   dtype=torch.float).to(self.device) # shape: (episode_length, state_dim
        dones = torch.tensor(transition_dict['dones'], 
                             dtype=torch.float).view(-1, 1).to(self.device) # shape: (episode_length, 1)
        rewards = (rewards + 8.0) / 8.0 # normalize rewards to [-1, 1].
                                        # The Pendulum env has a reward range of [-16.2736044, 0]
        # compute td error
        td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_targets - self.critic(states) # Temporal Difference Error(can also be called advantage in actor-critic)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device) # compute advantage using GAE
        # compute old log probabilities
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach()) # detach to avoid gradient flow to mu and std
        old_log_probs = action_dists.log_prob(actions) # for continuous distribution, log_prob returns the log of probability density function

        for _ in range(self.epochs):
            # compute new log probabilities
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions) 
            # compute importance weight
            ratio = torch.exp(log_probs - old_log_probs)
            # compute actor loss(PPO-clip loss) and critic loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() # PPO-clip loss. Minimize the negative of the PPO-clip loss
            critic_loss = F.mse_loss(self.critic(states), td_targets.detach()) # critic seeks to approximate td_targets
            # Update actor and critic networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def compute_advantage(self, gamma, lmbda, td_delta):
        'Using Generalized Advantage Estimation(GAE)'
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]: # reverse order
            advantage = delta + gamma * lmbda * advantage # Deltas that are more close to the epsode end will multiply more times gamma and lambda
            advantage_list.append(advantage)
        advantage_list.reverse() # reverse order back to original
        return torch.tensor(advantage_list, dtype=torch.float)