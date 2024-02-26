import sys

from env.cliff_walking_env import CliffWalkingEnv
from alg.policy_iteration import PolicyIteration
from alg.value_iteration import ValueIteration
from utils.arg import parse_args

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State Value: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("Policy: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
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
