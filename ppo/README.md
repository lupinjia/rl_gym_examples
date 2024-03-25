# Folder and File Explanation

- **alg**: Contains the implementation of PPO-Clip algorithm, including discrete and continuous action spaces.
- **cartpole.py**: PPO-Clip training example on the CartPole environment of OpenAI Gymnasium.
- **pendulum.py**: PPO-Clip training example on the Pendulum environment of OpenAI Gymnasium.

# Algorithm Introduction

Proximal Policy Optimization (PPO) is a state-of-the-art on-policy reinforcement learning algorithm developed by OpenAI. It is designed to balance the trade-off between performance and sample efficiency, addressing some of the key challenges faced by earlier policy gradient methods, such as the high variance in the policy gradient and the difficulty in tuning hyperparameters.

Though PPO kind of separates the behavior policy and the target policy, but because the difference between the two is small, it is still a on-policy algorithm.

PPO inherits the thought of TRPO, which intends to restrict the update step for avoiding the performance collapse of the agent. But it adopts a simpler and more computationally efficient approach.

You can find more information about PPO algorithm in the following resources:

- [Hands on RL](https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95)
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/chapter5/chapter5)

# Additional Notes

- The provided examples are designed for educational purposes, illustrating the core concepts of TD learning.
- Users are encouraged to modify and extend these examples to explore different aspects of TD learning and its applications.