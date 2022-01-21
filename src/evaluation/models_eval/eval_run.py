import sys
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVALMultiStep
from planner.action_policy import Policy

config_name = 'study_step_size'
val_run_name = config_name
# config_name = 'compare_epochs'
# val_run_name = 'epoch_50'

def main():
    eval_obj = MCEVALMultiStep(val_run_name)
    eval_obj.read_eval_config(config_name)
    eval_obj.policy = Policy()
    eval_obj.run()

if __name__=='__main__':
    main()
