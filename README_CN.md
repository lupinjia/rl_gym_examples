# rl_gym_examples

<img src="https://imgur.com/C7jhNYR.gif" width = "200" height = "70" alt="cliff_walking_gif"/>
<img src="https://imgur.com/s7uDhFJ.gif" width = "200" height = "90" alt="cliff_walking_gif"/>
<img src="https://imgur.com/Rf580ax.gif" width = "200" height = "100" alt="cliff_walking_gif"/>
<img src="https://imgur.com/QM8PfKs.gif" width = "200" height = "100" alt="cliff_walking_gif"/>

这个仓库包含了在OpenAI Gym环境使用Python实现的常见强化学习算法的示例代码。

这个仓库记录了我在学习过程中实现的RL算法，我希望它能帮助其他人更好地学习和理解RL算法。

## :sparkles: 特色

- 每个算法都有文档：每个文件夹都有一个README.md文件来介绍对应的算法
- 在OpenAI Gym环境中的示例
- 详细的注释

## :rocket: 开始使用

1. 首先，安装依赖，你可以使用conda或pip安装依赖：
    - conda (推荐)

    使用yml文件创建一个新的conda环境：
    ```Bash
    conda create -f rl_gym_examples.yml
    ```
    - pip

    你也可以使用pip安装依赖（尽管不推荐）：
    ```Bash
    pip install -r requirements.txt
    ```
    Python版本为3.8

2. 然后，你可以在对应的文件夹中运行示例，例如：
    ```Bash
    cd dp
    python gym_cliff_walking.py
    ```

### :bulb: 提示

依赖中的pytorch是CPU版本，你可以通过遵循[pytorch官网](https://pytorch.org/get-started/locally/)的指示来安装GPU版本。

## :books: 支持的算法

![RL Algorithm Development Path](https://i.imgur.com/Szbxpri.png) 

| 算法 | 观测空间 | 动作空间 | 基于模型或免模型 | 同策略或异策略 |
| --- | --- | --- | --- | --- |
| 动态规划（策略迭代或价值迭代） | 离散 | 离散 | 基于模型 | N/A |
| Sarsa | 离散 | 离散 | 免模型 | 同策略 |
| Q-learning | 离散 | 离散 | 免模型 | 异策略 |
| DQN | 连续 | 离散 | 免模型 | 异策略 |
| REINFORCE | 连续 | 离散/连续 | 免模型 | 同策略 |
| Actor-Critic | 连续 | 离散/连续 | 免模型 | 同策略 |
| TRPO/PPO | 连续 | 离散/连续 | 免模型 | 同策略 |
| DDPG | 连续 | 连续 | 免模型 | 异策略 |
| SAC | 连续 | 连续 | 免模型 | 异策略 |

## :file_folder: 文件结构

- 'dp': 动态规划
- 'td': 时序差分（TD）学习
- 'dqn': 深度Q网络（DQN）
- 'reinforce': REINFORCE算法（或Vanilla策略梯度）
- 'actor_critic': Actor-Critic算法
- 'ppo': 近端策略优化（PPO）算法
- 'ddpg': 深度确定性策略梯度（DDPG）算法
- 'sac': 软Actor-Critic（SAC）算法

## :memo: 参考文献

- [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL) 
- [Gymnasium](https://gymnasium.farama.org/) 
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/) 