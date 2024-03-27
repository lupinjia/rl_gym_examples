import sys

from env.cliff_walking_env import CliffWalkingEnv
from alg.policy_iteration import PolicyIteration
from alg.value_iteration import ValueIteration
from utils.arg import parse_args

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State Value: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Keep the output in 6 characters to make it easier to read.
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
    
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    is_policy_iteration = parse_args(argv)
    if is_policy_iteration:
        agent = PolicyIteration(env, theta, gamma)
        agent.policy_iteration()
    else:
        agent = ValueIteration(env, theta, gamma)
        agent.value_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
    

if __name__ == '__main__':
    main(sys.argv[1:])
