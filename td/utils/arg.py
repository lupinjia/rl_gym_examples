import sys
import getopt

def parse_args(argv):
    is_q_learning = False
    
    try:
        opts, _ = getopt.getopt(argv,"pq")
    except getopt.GetoptError:
        print('Error: arg.py -q')
        print('   or: arg.py')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-q':
            is_q_learning = True
    
    return is_q_learning