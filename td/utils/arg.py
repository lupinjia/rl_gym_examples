import sys
import getopt

def parse_args(argv):
    is_q_learning = False
    is_n_step = False
    is_dyna_q = False
    
    try:
        opts, _ = getopt.getopt(argv,"nqd")
    except getopt.GetoptError:
        print('Error: arg.py -q')
        print('   or: arg.py -n')
        print('   or: arg.py -d')
        print('   or: arg.py')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-q':
            is_q_learning = True
        elif opt == '-n':
            is_n_step = True
        elif opt == '-d':
            is_dyna_q = True
    
    return is_q_learning, is_n_step, is_dyna_q