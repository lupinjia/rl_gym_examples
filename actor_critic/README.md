# Folder and File Explanation

- **alg**: Contains the implementation of Actor-Critic algorithm, using temporal difference error.
- **cartpole.py**: Actor-Critic training example on the CartPole environment of OpenAI Gymnasium.

# Algorithm Introduction

Actor-Critic algorithms are a family of reinforcement learning algorithms that combine the actor and critic components. The actor component(a Neural Network) learns the optimal policy, while the critic component(another Neural Network) estimates the value function. The actor-critic algorithm uses the estimated value function to improve the policy by updating the policy based on the temporal difference error.

You can find more information about Actor-Critic algorithm in the following resources:

- [Hands on RL](https://hrl.boyuai.com/chapter/2/actor-critic%E7%AE%97%E6%B3%95)
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/chapter9/chapter9)

# Additional Notes

- The provided examples are designed for educational purposes, illustrating the core concepts of TD learning.
- Users are encouraged to modify and extend these examples to explore different aspects of TD learning and its applications.