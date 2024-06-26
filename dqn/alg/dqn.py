import collections
import random
import numpy as np
import torch
from torch.nn import functional as F


class ReplayBuffer:
    ''' Experiences Replay Buffer '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # double ended queue, FIFO

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample data from buffer, with size of batch_size
        transitions = random.sample(self.buffer, batch_size) # randomly sample batch_size transitions from buffer
        # return a list of transitions, each transition is a tuple of (state, action, reward, next_state, done)
        state, action, reward, next_state, done = zip(*transitions) # unpack the list, merge the elements in transitions with same index into a new tuple.
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' Q Network
     only one hidden layer, ReLU activation function, no dropout, no batch normalization. '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update  # target net update frequency
        self.count = 0                      # update count
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):  # epsilon-greedy policy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)  # view to (1, state_dim)
            action = self.q_net(state).argmax().item()  # select action with max Q value
        return action
    
    def max_q_value(self, state):
        # state: 1 state
        state = torch.tensor(state, dtype=torch.float, device=self.device).view(1, -1)
        return self.q_net(state).max().item() # 直接max()就可以得到标量

    def update(self, transition_dict):
        # experience data of a batch
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).view(-1, self.state_dim).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).view(-1, self.state_dim).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions) # 遵从gather()的规则取值
                                                         # q_values会变成与actions一样的形状:(batch_size, 1)
                                                         # 相当于取出actions对应的Q值.
        # max Q value of next state, for calcing td target
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1) # max(1)[1]表示在dim=1上取最大值后得到其索引.
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action) # 选择max_action对应的Q值.
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1) # 在dim=1(action维度)上求最大值.
                       # 调用max(1)后返回的是torch.return_types.max(values, indices)类型, 下标0提取values, 即最大值.提取出来是一维tensor.
                       # 再用view()将一维tensor转换为二维tensor, 维度为(batch_size, 1).
        
        # 如果下一个状态为终止状态(terminated), 则不需要考虑下一个状态的Q值.
        # 但truncated需要考虑
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # Q network updates every step, target network updates every target_update steps.
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # copy q_net weights to target_q_net
        self.count += 1