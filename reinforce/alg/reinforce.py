import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[i], action_dim))
            else:
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.ReLU())
        layers.append(F.softmax(dim=1))  # discrete action, use softmax as output activation
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        layers = []
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
    
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device, action_type='discrete'):
        self.action_type = action_type
        if action_type == 'discrete':
            self.policy_net = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        elif action_type == 'continuous':
            self.policy_net = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):  # randomly sample an action according to the probability distribution
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.device) # 将state再包一层list后转为tensor
                                                                         # 这样就可以得到1*state_dim的tensor
                                                                         # 但这个方法会引起警告UserWarning: Creating a tensor from a list 
                                                                         # of numpy.ndarrays is extremely slow. Please consider converting the
                                                                         # list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        if self.action_type == 'discrete':
            probs = self.policy_net(state)
            action_dist = torch.distributions.Categorical(probs) # Categorical Distribution(https://en.wikipedia.org/wiki/Categorical_distribution)
                                                                # Categorical是多项分布的特例, 相当于只进行1次试验的多项分布
            action = action_dist.sample() # Categorical采样时会根据概率分布在[0,n-1]之间采样一个动作
            return action.item()  # discrete action, return scalar
        else:
            mu, std = self.policy_net(state)
            action_dist = torch.distributions.Normal(mu, mu*0.0 + std)  # Gaussian Distribution
            action = action_dist.sample()
            return [action.item()]  # continuous action, return vector

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # Back to Front, calc G
            reward = reward_list[i]
            state = torch.tensor(state_list[i].reshape(1, -1),
                                 dtype=torch.float).to(self.device)
            action = torch.tensor(action_list[i]).view(-1, 1).to(self.device)
            if self.action_type == 'discrete':
                log_prob = torch.log(self.policy_net(state).gather(1, action))  # calc log probability of action
            else:
                mu, std = self.policy_net(state)
                action_dist = torch.distributions.Normal(mu, std)
                log_prob = action_dist.log_prob(action)
            G = self.gamma * G + reward  # calculate return starting from this step
            loss = -log_prob * G         # loss of each step
            loss.backward()              # The gradient will be accumulated in each step
        self.optimizer.step()