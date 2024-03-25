# Folder and File Explanation

- **alg**: Contains the implementation of DQN algorithms, including DQN, Double DQN, and Dueling DQN.
- **cartpole.py**: DQN training example on the CartPole environment of OpenAI Gymnasium.
- **pendulum.py**: DQN training example on the Pendulum environment of OpenAI Gymnasium. Note that the Pendulum env has continuous action space, so the DQN algorithm used here adopts a action discretization technique to handle the continuous action space.

# Algorithm Introduction

Deep Q-Network (DQN) is a off-policy reinforcement learning algorithm that combines the ideas of Q-learning, a model-free method for learning action-value pairs, with deep neural networks to approximate the action-value function (also known as the Q-function). Developed by researchers at DeepMind, DQN has been particularly successful in playing complex video games and other tasks that require learning from high-dimensional sensory input.

The key features of DQN are:

- **Experience replay**: DQN uses a replay buffer to store and sample experiences from the environment. This allows the agent to learn from a wide range of experiences, rather than just the most recent ones.
- **Target network**: DQN uses a target network to stabilize the training process. The target network is a copy of the online network that is updated less frequently than the online network.

You can find more information about DQN in the following resources:

- [Hands on RL](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)

# Additional Notes

- The provided examples are designed for educational purposes, illustrating the core concepts of TD learning.
- Users are encouraged to modify and extend these examples to explore different aspects of TD learning and its applications.