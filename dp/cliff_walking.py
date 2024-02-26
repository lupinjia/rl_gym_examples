import gymnasium as gym
import sys

from alg.policy_iteration import PolicyIteration
from alg.value_iteration import ValueIteration
from utils.arg import parse_args

def main(argv):
    # cliff walking env: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make('CliffWalking-v0', render_mode='human')
    print("wrapped env:", env)
    env = env.unwrapped # unwrap to access the state transition function
    obs, info = env.reset()
    print("unwrapped env:", env)
    env.render()
    # find the optimal policy using policy iteration or value iteration
    # dynamic programming does not require agent to interact with the environment.
    theta = 1e-5
    gamma = 0.9
    is_policy_iteration = parse_args(argv)
    if is_policy_iteration:
        agent = PolicyIteration(env, theta, gamma)
        agent.policy_iteration()
    else:
        agent = ValueIteration(env, theta, gamma)
        agent.value_iteration()
    # show the optimal policy
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break

if __name__ == '__main__':
    main(sys.argv[1:])

