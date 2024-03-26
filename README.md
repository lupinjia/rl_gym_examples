# rl_gym_examples

This repository contains examples of common Reinforcement Learning algorithms in openai gymnasium environment, using Python.

This repo records my implementation of RL algorithms while learning, and I hope it can help others learn and understand RL algorithms better.



## Features

- Document for each algorithm: Every folder has a README.md file to introduce the algorithm
- Examples in OpenAI Gymnasium environments
- Detailed comments

## Dependencies

- gymnasium 0.29.1
- gymnasium[toy-text]
- tqdm
- numpy
- matplotlib
- pytorch

## Getting Started

### Conda(Recommended)

First, create a new conda environment with Python 3.8 and activate it:
```Bash
conda create -n rl_gym_examples python=3.8
conda activate rl_gym_examples
```

Then, import the yml file to install the dependencies:
```Bash
conda env import -f rl_gym_examples.yml
```
### Pip

You can also install the dependencies using pip(though it is not recommended):
```Bash
pip install -r requirements.txt
```

### Tips

The pytorch in the dependencies is cpu version, you can install the gpu version by following the instructions in the [pytorch website](https://pytorch.org/get-started/locally/).

## Supported Algorithms

![RL Algorithm Development Path](https://i.imgur.com/Szbxpri.png)

| Algorithm | Observation Space | Action Space | Model-based or Model-free | On-policy or Off-policy |
| --- | --- | --- | --- | --- |
| Dynamic Programming(Policy Iteration or Value Iteration) | Discrete | Discrete | Model-based | NA |
| Sarsa | Discrete | Discrete | Model-free | on-policy |
| Q-learning | Discrete | Discrete | Model-free | off-policy |
| DQN | Continuous | Discrete | Model-free | off-policy |
| REINFORCE | Continuous | Discrete/Continuous | Model-free | on-policy |
| Actor-Critic | Continuous | Discrete/Continuous | Model-free | on-policy |
| TRPO/PPO | Continuous | Discrete/Continuous | Model-free | on-policy |
| DDPG | Continuous | Continuous | Model-free | off-policy |
| SAC | Continuous | Continuous | Model-free | off-policy |

## File Structure

- 'dp':  Dynamic Programming
- 'td':  Temporal Difference (TD) learning
- 'dqn': Deep Q Network (DQN)
- 'reinforce': REINFORCE algorithm(or Vanilla Policy Gradient)
- 'actor_critic': Actor-Critic algorithm
- 'ppo': Proximal Policy Optimization (PPO) algorithm
- 'ddpg': Deep Deterministic Policy Gradient (DDPG) algorithm
- 'sac': Soft Actor-Critic (SAC) algorithm

## References

- [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)
- [Gymnasium](https://gymnasium.farama.org/)
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/)