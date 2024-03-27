import sys
import getopt

def parse_args(argv):
    '''parse arguments to determine whether to use policy iteration or value iteration'''
    is_policy_iteration = False
    
    try:
        opts, _ = getopt.getopt(argv,"p")
    except getopt.GetoptError:
        print('Error: arg.py -p')
        print('   or: arg.py')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-p':
            is_policy_iteration = True
    
    return is_policy_iteration