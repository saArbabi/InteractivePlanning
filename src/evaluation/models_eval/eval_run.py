import sys
sys.path.insert(0, './src')
from evaluation.eval_data_obj import EvalDataObj
from evaluation.eval_obj import MCEVALMultiStep
from evaluation.eval_obj_singlestep import MCEVALSingleStep
import json
import pickle


# config_name = 'study_seq_len'
# config_name = 'cae_008'
config_name = 'test'
# config_name = 'lstm_and_mlp'
val_run_name = config_name
# config_name = 'compare_epochs'
# val_run_name = 'epoch_50'
eval_config_dir = './src/evaluation/models_eval/'+ config_name +'.json'

def read_eval_config(config_name):
    with open(eval_config_dir, 'rb') as handle:
        eval_config = json.load(handle)
    return eval_config


def loadScalers():
    with open('./src/datasets/'+'state_scaler', 'rb') as f:
        state_scaler = pickle.load(f)
    with open('./src/datasets/'+'action_scaler', 'rb') as f:
        action_scaler = pickle.load(f)
    return state_scaler, action_scaler

def main():
    eval_config = read_eval_config(config_name)
    data_obj = EvalDataObj()
    states_arr, targets_arr = data_obj.load_val_data()
    state_scaler, action_scaler = loadScalers() # will set the scaler attributes

    model_names = eval_config['model_map'].keys()
    for model_name in model_names:
        model_type, _, traffic_density = eval_config['model_map'][model_name]

        if model_type == 'CAE':
            eval_obj = MCEVALMultiStep(eval_config, val_run_name)

        if model_type == 'MLP' or model_type == 'LSTM':
            eval_obj = MCEVALSingleStep(eval_config, val_run_name)

        eval_obj.eval_config_dir = eval_config_dir
        eval_obj.episode_ids = data_obj.load_test_episode_ids(traffic_density)
        eval_obj.states_arr, eval_obj.targets_arr = states_arr, targets_arr
        eval_obj.state_scaler, eval_obj.action_scaler = state_scaler, action_scaler
        eval_obj.run(model_name)

if __name__=='__main__':
    main()
