import sys
import getopt

def parse_args(argv):
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