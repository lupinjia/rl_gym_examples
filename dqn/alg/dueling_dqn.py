import torch
from torch.nn import functional as F
import collections
import random
import numpy as np

class VAnet(torch.nn.Module):
    '''
    只有一层隐藏层的A网络和V网络
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim) # 输出优势值
        self.fc_V = torch.nn.Linear(hidden_dim, 1) # 输出状态值函数
    
    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1) # 状态值函数加上优势值减去平均优势值
        return Q

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 隐藏层1
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim) # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x) # 输出层不使用激活函数
    
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size) # 从buffer中随机采样batch_size个数据,返回一个元素为元组的列表.每个数据为一个元组.
        state, action, reward, next_state, done = zip(*transitions) # 将最外层的列表解包，用zip函数将每个元组相同位置的元素组合成一个新的元组.
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class DQN:
    '''
    DQN算法, 包括Double DQN and Dueling DQN
    '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device, 
                 dqn_type='VanillaDQN'):
        # Construct Q-network and target network
        if dqn_type == "DuelingDQN": # Dueling DQN adopts different network architecture
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        else: # Double DQN
            self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # params
        self.action_dim = action_dim
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type
    
    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item() # 选择Q值最大的动作
        return action

    def max_q_value(self, state):
        # 输入的state为单个样本的数据
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        return self.q_net(state).max().item() # 直接max()就可以得到标量
    
    def update(self, transition_dict):
        # 一个batch的经验数据
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions) # 遵从gather()的规则取值
                                                         # q_values会变成与actions一样的形状:(batch_size, 1)
                                                         # 相当于取出actions对应的Q值.
        # 下个状态的最大Q值
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
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        # Q网络每个step都会更新. 目标网络每target_update个step更新一次.
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络, 将Q网络参数复制到目标网络中
        self.count += 1
