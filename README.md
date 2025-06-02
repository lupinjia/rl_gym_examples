# rl_gym_examples
<table>
  <tr>
    <td><img src="https://imgur.com/C7jhNYR.gif" width = "200" height = "70" alt="cliff_walking_gif"/></td>
    <td><img src="https://imgur.com/s7uDhFJ.gif" width = "200" height = "90" alt="cartpole_gif"/></td>
  </tr>
  <tr>
    <td><img src="https://imgur.com/Rf580ax.gif" width = "200" height = "100" alt="bipedal_walker_gif"/></td>
    <td><img src="https://imgur.com/QM8PfKs.gif" width = "200" height = "100" alt="inverted_pendulum_gif"/></td>
  </tr>
</table>

This repository contains examples of common Reinforcement Learning algorithms in openai gymnasium environment, using Python.

This repo records my implementation of RL algorithms while learning, and I hope it can help others learn and understand RL algorithms better.


## :sparkles:Features

- Document for each algorithm: Every folder has a README.md file to introduce the algorithm
- Examples in OpenAI Gymnasium environments
- Detailed comments

## :rocket:Getting Started

### :package:Choose Version

Choose the version you want to use:
- [Simple Implementation]: The simplest implementation of each algorithm, showing the core logic of the algorithm.

### :computer:Prepare the Environment & Install Dependencies

1. clone this repository
   
   ```Bash
   git clone git@github.com:lupinjia/rl_gym_examples.git
   ```

2. create a new virtual environment with python 3.8
   
   ```Bash
   conda create -n env_name python=3.8
   ```

3. install the dependencies, you can install dependencies using pip:
    ```Bash
    pip install swig
    pip install -r requirements.txt
    ```

4. Then, you can run the examples in the corresponding folders, for example:
   
    ```Bash
    cd dp
    python gym_cliff_walking.py
    ```

### :bulb:Tips

The pytorch in the dependencies is cpu version by default, you can install the gpu version by following the instructions in the [pytorch website](https://pytorch.org/get-started/locally/).

## :books:Supported Algorithms

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

## :file_folder:File Structure

- 'dp':  Dynamic Programming
- 'td':  Temporal Difference (TD) learning
- 'dqn': Deep Q Network (DQN)
- 'reinforce': REINFORCE algorithm(or Vanilla Policy Gradient)
- 'actor_critic': Actor-Critic algorithm
- 'ppo': Proximal Policy Optimization (PPO) algorithm
- 'ddpg': Deep Deterministic Policy Gradient (DDPG) algorithm
- 'sac': Soft Actor-Critic (SAC) algorithm

## :memo:References

- [Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)
- [Gymnasium](https://gymnasium.farama.org/)
- [EasyRL](https://datawhalechina.github.io/easy-rl/#/)