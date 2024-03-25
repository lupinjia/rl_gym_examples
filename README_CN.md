# rl_examples

这个仓库包含了在OpenAI Gymnasium环境使用Python实现的常见强化学习算法示例。

这个仓库记录了我在学习RL算法过程中实现的代码，我希望它能帮助其他人更好地学习和理解RL算法。

## TODO

- [x] 给每个文件夹添加readme文件，介绍算法
- [ ] 完善每个算法的注释

## 特色

- 每个算法配有文档：每个文件夹都有一个README.md文件来解释算法
- 在OpenAI Gym环境中的示例
- 详细的注释

## 依赖

- gymnasium
- gymnasium[toy-text]
- tqdm
- numpy
- matplotlib
- pytorch

## 支持的算法

![强化学习算法发展路径](https://i.imgur.com/Szbxpri.png)

| 算法 | 观测空间 | 动作空间 | 基于模型或无模型 | 同策略或异策略 |
| --- | --- | --- | --- | --- |
| 动态规划（策略迭代或值迭代） | 离散 | 离散 | 基于模型 | NA |
| Sarsa | 离散 | 离散 | 无模型 | 同策略 |
| Q-learning | 离散 | 离散 | 无模型 | 异策略 |
| DQN | 连续 | 离散 | 无模型 | 异策略 |
| REINFORCE | 连续 | 离散/连续 | 无模型 | 同策略 |
| Actor-Critic | 连续 | 离散/连续 | 无模型 | 同策略 |
| TRPO/PPO | 连续 | 离散/连续 | 无模型 | 同策略 |
| DDPG | 连续 | 连续 | 无模型 | 异策略 |
| SAC | 连续 | 连续 | 无模型 | 异策略 |

## 文件结构

- 'dp': 动态规划
- 'td': 时序差分（TD）学习
- 'dqn': 深度Q网络（DQN）
- 'reinforce': REINFORCE算法（或称为普通策略梯度）
- 'actor_critic': 演员-评论家（Actor-Critic）算法
- 'ppo': 近端策略优化（PPO）算法
- 'ddpg': 深度确定性策略梯度（DDPG）算法
- 'sac': 软演员-评论家（SAC）算法

## 参考文献

- [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)
- [Gymnasium](https://gymnasium.farama.org/)
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/)