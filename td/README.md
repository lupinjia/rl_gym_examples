# Folder and File Explanation

- **alg**: Contains the implementation of TD learning algorithms, including SARSA (State-Action-Reward-State-Action), Q-learning, N-step SARSA, and Dyna-Q.
- **utils**: Houses utility functions for the TD learning examples, like argument parser.
- **blackjack.py**: An example script showing how to train a TD agent on the Blackjack environment from OpenAI Gymnasium.
- **cliff_walking.py**: An example script showing how to train a TD agent on the CliffWalking environment from OpenAI Gym.
- **Frozen_lake.py**: An example script showcasing the training process of a TD agent on the FrozenLake environment.
- **taxi.py**: An example script showcasing the training process of a TD agent on the Taxi environment.

# Algorithm Introduction

Temporal Difference (TD) learning is a model-free method used in reinforcement learning to estimate the value function and the optimal policy. Unlike DP, which is model-based, TD learning does not require complete knowledge of the environment's dynamics. TD learning combines the bootstrpping of DP and sampling of Monte Carlo(MC), showing best performance among three methods. TD learning relies on the **Bellman Expectation Equation** for its updates.

You can find more information about TD in the following resources:

- [Hands on RL](https://hrl.boyuai.com/chapter/1/%E6%97%B6%E5%BA%8F%E5%B7%AE%E5%88%86%E7%AE%97%E6%B3%95)

# Additional Notes

- The provided examples are designed for educational purposes, illustrating the core concepts of TD learning.
- Users are encouraged to modify and extend these examples to explore different aspects of TD learning and its applications.