import gymnasium as gym
import sys

from alg.policy_iteration import PolicyIteration
from alg.value_iteration import ValueIteration
from utils.arg import parse_args

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State Value: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Keep the output in 6 characters, to make it more readable
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("Policy: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Some special states, like the cliff in Cliff-Walking
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # goal state
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

def main(argv):
    # cliff walking env: https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make('FrozenLake-v1', render_mode='human')
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
    action_meaning = ['<', 'v', '>', '^']
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
    # show the optimal policy
    # due to the slippery nature of the frozen lake, the agent may not reach the goal state in each episode, even if it is optimal.
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break

if __name__ == '__main__':
    main(sys.argv[1:])

