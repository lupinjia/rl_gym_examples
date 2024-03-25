# Folder and File Explanation

- **alg**: algorithm folder, consisting of implementation of dynamic programming algorithms, including policy iteration and value iteration.
- **env**: self-made environment, including a simple cliff_walking environment, which can help understand the envs in openai gym.
- **utils**: some utility functions, including argument parser.
- **video**: some videos of the algorithm running results.
- **frozen_lake.py**: training example on the frozen_lake environment of openai gymnasium.
- **gym_cliff_walking.py**: training example on the cliff_walking environment of openai gym.
- **self_cliff_walking.py**: training example on the self-made cliff_walking environment.

# Algorithm Introduction

Dynamic Programming (DP) is a model-based method to solve Markov Decision Process (MDP). It is **model-based** so it needs the information of the environment, including state transition probabilities and rewards. It utilizes the environment information to derive state value and action value, through **Bellman Expectation Equation** or **Bellman Optimality Equation**. One of the most important characteristics of DP is **bootstrapping**, because it calculates the state value using state value. 

You can find more information about DP in the following resources:

- [Hands on RL](https://hrl.boyuai.com/chapter/1/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%AE%97%E6%B3%95)
