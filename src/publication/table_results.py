import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.planner.state_indexs import StateIndxs
indxs = StateIndxs()

np.set_printoptions(suppress=True)

def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_scenario_err(veh_axis, state_axis, model_run_name):
    """
    Input shpae [m scenarios, n traces, steps_n, state_index]
    Return shape [m, steps_n]
    """
    posx_true = true_collections[model_run_name][:,:,19:, veh_axis[state_axis]+2]
    posx_pred = pred_collections[model_run_name][:,:,:, veh_axis[state_axis]]
    _pred = np.zeros([posx_true.shape[0], 50, 21])
    _true = np.zeros([posx_true.shape[0], 1, 21])
    for i in range(1, 21):
        _true[:, :, i] = _true[:, :, i-1] + posx_true[:, :, i-1]*0.1
        _pred[:, :, i] = _pred[:, :, i-1] + posx_pred[:, :, i-1]*0.1

    posx_true = _true
    posx_pred = _pred

    scenario_err_arr = []
    for m in range(posx_true.shape[0]):
        scenario_err_arr.append(get_trace_err(posx_pred[m, :, :], posx_true[m, :, :]))
    return np.array(scenario_err_arr)

def get_rwse(scenario_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(scenario_err_arr, axis=0)**0.5

# %%
""" effect of guided learning """
mc_run_name = ['all_density']

model_val_run_map = {
    # 'cae_003': mc_run_name, #
    # 'cae_010': mc_run_name, # "allowed_error": 0.1
    'cae_023': mc_run_name, # "allowed_error": 0.2
    # 'cae_012': mc_run_name, # "allowed_error": 0.3
    'cae_024': mc_run_name, # "allowed_error": 0.4
    }
# %%
""" effect of using different architectures """
mc_run_name = ['all_density']

model_val_run_map = {
    # 'cae_003': mc_run_name, #
    # 'cae_008': mc_run_name, #
    'cae_026': mc_run_name, # single decoder
    }

# %%
""" compare CAE, MLP and LSTM """
mc_run_name = ['all_density']

model_val_run_map = {
    'cae_018': mc_run_name, #
    'mlp_002': mc_run_name, #
    'lstm_002': mc_run_name, #
    }

# %%
""" No response dynamics considered """
mc_run_name = ['all_density']

model_val_run_map = {
    'cae_025': mc_run_name, #
    }
# %%

true_collections = {}
pred_collections = {}
for model_name in list(model_val_run_map.keys()):
    mc_run_names = model_val_run_map[model_name]
    for mc_run_name in mc_run_names:
        model_run_name = model_name+'_'+mc_run_name
        exp_dir = './src/models/experiments/'+model_name+'/' + mc_run_name

        with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
            true_collections[model_run_name] = np.array(pickle.load(handle))

        with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
            pred_collections[model_run_name] = np.array(pickle.load(handle))

true_collections[model_run_name].shape
pred_collections[model_run_name].shape
model_run_names = list(true_collections.keys())
model_run_names

# %%
time_steps = np.linspace(0, 2., 21)
veh_axiss = [indxs.indx_m, indxs.indx_y, indxs.indx_f, indxs.indx_fadj]
veh_names = ['veh_m', 'veh_y', 'veh_f', 'veh_fadj']
long_err_at_2s = {}
lat_err_at_2s = {}

for model_run_name in model_run_names:
    long_err_at_2s[model_run_name] = []
    lat_err_at_2s[model_run_name] = []

for veh_axis in veh_axiss:
    """
    rwse long_speed
    """
    long_err_collections = {}
    for model_run_name in model_run_names:
        long_err_collections[model_run_name] = get_scenario_err(veh_axis, 'vel', model_run_name)
    for model_run_name in model_run_names:
        scenario_err_arr = long_err_collections[model_run_name]
        error_total = get_rwse(scenario_err_arr)
        long_err_at_2s[model_run_name].append(error_total[-1].round(2))

    """
    rwse lat_speed
    """
    lat_err_collections = {}
    for model_run_name in model_run_names:
        lat_err_collections[model_run_name] = get_scenario_err(veh_axis, 'act_lat', model_run_name)

    for model_run_name in model_run_names:
        scenario_err_arr = lat_err_collections[model_run_name]
        error_total = get_rwse(scenario_err_arr)
        lat_err_at_2s[model_run_name].append(error_total[-1].round(2))

print(long_err_at_2s)
print(lat_err_at_2s)
