import sys
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVAL
from planner.action_policy import Policy

val_run_name = 'test_2'
def main():
    eval_obj = MCEVAL(val_run_name)
    eval_obj.policy = Policy()
    eval_obj.run()

if __name__=='__main__':
    main()
