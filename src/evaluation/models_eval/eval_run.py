import sys
sys.path.insert(0, './src')
from models.evaluation.eval_obj import MCEVAL
val_run_name = 'data_033_1'
def main():
    eval_obj = MCEVAL(val_run_name)
    eval_obj.run()

if __name__=='__main__':
    main()
