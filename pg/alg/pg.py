import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        # Construct neural network according to input and output dimensions, hidden dim
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
    
    def forward(self, x):
        return self.net(x)

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
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
        self.std = nn.Parameter(0.1 * torch.ones(action_dim))
    
    def forward(self, x):
        mean = self.net(x)
        return mean, self.std

class PolicyGradientDiscrete:
    def __init__(self, state_dim, hidden_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu'):
        # Initialize policy network and optimizer
        self.policy_net = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_dist = torch.distributions.Categorical(torch.tensor([1/action_dim for _ in range(action_dim)]))  # init a equal prob distribution
        self.gamma = gamma
        self.device = device
    
    def update_dist(self, action_probs):
        # Update action distribution based on action probabilities
        self.action_dist = torch.distributions.Categorical(action_probs)
    
    def select_action(self, state):
        # Select action based on current policy network
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        action_probs = self.policy_net(state)
        self.update_dist(action_probs)
        action = self.action_dist.sample()
        return action.item()  # discrete action, return int scalar
    
    def learn(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # Back to Front, calc G
            reward = reward_list[i]
            state = torch.tensor(state_list[i],
                                 dtype=torch.float).view(1, -1).to(self.device)
            action = torch.tensor(action_list[i]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))  # calc log probability of action
            G = self.gamma * G + reward  # calculate return starting from this step
            loss = -log_prob * G         # loss of each step
            loss.backward()              # The gradient will be accumulated in each step
        self.optimizer.step()

class PolicyGradientContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu'):
        # Initialize policy network and optimizer
        self.policy_net = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_dist = torch.distributions.Normal(0, 0.1)  # init a normal distribution
        self.gamma = gamma
        self.device = device
    
    def update_dist(self, mean, std):
        # Update action distribution based on action probabilities
        self.action_dist = torch.distributions.Normal(mean, mean*0. + std)
    
    def select_action(self, state):
        # Select action based on current policy network
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        mean, std = self.policy_net(state)
        self.update_dist(mean, std)
        action = self.action_dist.sample()
        return action.numpy()
    
    def learn(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # Back to Front, calc G
            reward = reward_list[i]
            state = torch.tensor(state_list[i],
                                 dtype=torch.float).view(1, -1).to(self.device)
            action = torch.tensor(action_list[i]).view(-1, 1).to(self.device)
            mu, std = self.policy_net(state)
            self.update_dist(mu, std)
            log_prob = self.action_dist.log_prob(action)
            G = self.gamma * G + reward  # calculate return starting from this step
            loss = -log_prob * G         # loss of each step
            loss.backward()              # The gradient will be accumulated in each step
        self.optimizer.step()
