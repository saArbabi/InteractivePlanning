import sys
sys.path.insert(0, './src')
from models.evaluation.eval_obj import MCEVAL
from src.planner.action_policy import Policy

val_run_name = 'test_1'
def main():
    eval_obj = MCEVAL(val_run_name)
    eval_obj
    eval_obj.run()

if __name__=='__main__':
    main()
