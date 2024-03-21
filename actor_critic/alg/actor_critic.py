# Based on the code of Hands-on-RL(https://github.com/boyu-ai/Hands-on-RL/blob/main/)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module): # 策略网络, 使用PolicyGradient
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # 经过softmax转换为动作的概率

class ValueNet(nn.Module): 
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) # 状态s下对应的值函数为标量, 所以输出维度固定为1
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x) # 输出值函数

class ActorCritic: # 策略网络和值函数网络的结合
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        # 策略网络和值函数网络的初始化
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 策略网络和值函数网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # hyperparameters
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state) # 得到动作概率
        action_dist = torch.distributions.Categorical(probs) # 得到动作分布
        action = action_dist.sample() # 采样动作
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states) # 计算时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_error.detach()) # detach()防止梯度传递, 共享数据内存, 值相同
        # 值函数网络的损失函数, 均方误差
        # ?mean需要吗
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        # prepare to update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # calc gradients
        actor_loss.backward()
        critic_loss.backward()
        # update weights
        self.actor_optimizer.step()
        self.critic_optimizer.step()