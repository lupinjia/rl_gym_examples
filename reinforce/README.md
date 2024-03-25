# Folder and File Explanation

- **alg**: Contains the implementation of REINFORCE algorithm, or the Vanilla Policy Gradient algorithm.
- **cartpole.py**: REINFORCE training example on the CartPole environment of OpenAI Gymnasium.

# Algorithm Introduction

REINFORCE is a **policy gradient** algorithm in the field of reinforcement learning (RL). It is a Monte Carlo method that uses complete episodes of experience to estimate the gradient of the performance function with respect to the policy parameters. This algorithm is designed to learn a **stochastic policy**, where the policy outputs a probability distribution over actions given the current state.

You can find more information about REINFORCE algorithm and Policy Gradient in the following resources:

- [Hands on RL](https://hrl.boyuai.com/chapter/2/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95)
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/chapter4/chapter4?id=_41-%e7%ad%96%e7%95%a5%e6%a2%af%e5%ba%a6%e7%ae%97%e6%b3%95)

# Additional Notes

- The provided examples are designed for educational purposes, illustrating the core concepts of TD learning.
- Users are encouraged to modify and extend these examples to explore different aspects of TD learning and its applications.