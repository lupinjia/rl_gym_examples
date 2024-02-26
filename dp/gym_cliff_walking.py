import gymnasium as gym
import sys

from alg.policy_iteration import PolicyIteration

def main():
    # cliff walking env: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make('CliffWalking-v0', render_mode='human')
    print("wrapped env:", env)
    env = env.unwrapped # unwrap to access the state transition function
    obs, info = env.reset()
    print("unwrapped env:", env)
    env.render()
    # find the optimal policy using policy iteration.
    # dynamic programming does not require agent to interact with the environment.
    theta = 1e-5
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    # show the optimal policy
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break

if __name__ == '__main__':
    main()

