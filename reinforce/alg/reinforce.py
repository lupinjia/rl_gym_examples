import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__() # inherit from nn.Module
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # dim=1表示在action维度上求和计算softmax
    
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device) # 将state再包一层list后转为tensor
                                                                         # 这样就可以得到1*state_dim的tensor
                                                                         # 但这个方法会引起警告UserWarning: Creating a tensor from a list 
                                                                         # of numpy.ndarrays is extremely slow. Please consider converting the
                                                                         # list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs) # Categorical分布
                                                             # Categorical是多项分布的特例, 相当于只进行1次试验的多项分布
        action = action_dist.sample() # Categorical采样时会根据概率分布在[0,n-1]之间采样一个动作
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            # print("update state shape: ", state.shape)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action)) # 计算采取的动作的对数概率
            G = self.gamma * G + reward # 计算累积折扣奖励, 使用当前动作往后的奖励
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降