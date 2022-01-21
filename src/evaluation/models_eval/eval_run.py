import sys
sys.path.insert(0, './src')

# config_name = 'study_step_size'
config_name = 'test'
val_run_name = config_name
model_type = 'CAE'
model_type = 'MLP'
# config_name = 'compare_epochs'
# val_run_name = 'epoch_50'

def main():
    if model_type == 'CAE':
        from evaluation.eval_obj import MCEVALMultiStep
        eval_obj = MCEVALMultiStep(val_run_name)

    if model_type == 'MLP' or model_type == 'LSTM':
        from evaluation.eval_obj_singlestep import MCEVALSingleStep
        eval_obj = MCEVALSingleStep(val_run_name)

    eval_obj.read_eval_config(config_name)
    eval_obj.run()

if __name__=='__main__':
    main()
