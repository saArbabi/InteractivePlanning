import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.planner.state_indexs import StateIndxs
indxs = StateIndxs()

np.set_printoptions(suppress=True)
# %%
""" effect of guided learning """
val_run_name = ['all_density']

model_val_run_map = {
    'cae_003': val_run_name, #
    'cae_010': val_run_name, # "allowed_error": 0.1
    'cae_011': val_run_name, # "allowed_error": 0.2
    'cae_012': val_run_name, # "allowed_error": 0.3
    }
# %%
""" effect of using different architectures """
val_run_name = ['all_density']

model_val_run_map = {
    # 'cae_003': val_run_name, #
    'cae_008': val_run_name, #
    'cae_009': val_run_name, #
    }

# %%
""" compare CAE, MLP and LSTM """
val_run_name = ['all_density']

model_val_run_map = {
    'cae_012': val_run_name, #
    'mlp_001': val_run_name, #
    'lstm_001': val_run_name, #
    }
# %%
true_collections = {}
pred_collections = {}
for model_name in list(model_val_run_map.keys()):
    val_run_names = model_val_run_map[model_name]
    for val_run_name in val_run_names:
        model_run_name = model_name+'_'+val_run_name
        exp_dir = './src/models/experiments/'+model_name+'/' + val_run_name

        with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
            true_collections[model_run_name] = np.array(pickle.load(handle))

        with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
            pred_collections[model_run_name] = np.array(pickle.load(handle))

true_collections[model_run_name].shape
pred_collections[model_run_name].shape
model_run_names = list(true_collections.keys())
model_run_names

# %%
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

    scenario_err_arr = []
    for m in range(posx_true.shape[0]):
        scenario_err_arr.append(get_trace_err(posx_pred[m, :, :], posx_true[m, :, :]))
    return np.array(scenario_err_arr)

def get_rwse(scenario_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(scenario_err_arr, axis=0)**0.5

# %%
time_steps = np.linspace(0, 2., 21)
veh_axiss = [indxs.indx_m, indxs.indx_y, indxs.indx_f, indxs.indx_fadj]
veh_axis = veh_axiss[0]
"""
rwse long_speed
"""
fig = plt.figure(figsize=(6, 4))
long_speed = fig.add_subplot(211)
fig.subplots_adjust(hspace=0.1)

long_speed_err_collections = {}
for model_run_name in model_run_names:
    long_speed_err_collections[model_run_name] = get_scenario_err(veh_axis, 'vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_speed_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    long_speed.plot(time_steps, error_total, label=model_run_name)

long_speed.set_ylabel('RWSE Long.speed ($ms^{-1}$)', labelpad=10)
"""
rwse lat_speed
"""
lat_speed = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.5)

lat_speed_err_collections = {}
for model_run_name in model_run_names:
    lat_speed_err_collections[model_run_name] = get_scenario_err(veh_axis, 'act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_speed_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    lat_speed.plot(time_steps, error_total, label=model_run_name)
lat_speed.set_ylabel('RWSE Lat.speed ($ms^{-1}$)')
lat_speed.set_xlabel('Time horizon (s)')
lat_speed.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)
# %%
for model_run_name in model_run_names:
    err_at_2s = long_speed_err_collections[model_run_name][:, -1].mean()**0.5
    # print(model_run_name+' err_at_2s: ', err_at_2s)
    print(round(err_at_2s, 2))

# %%
# %%
for model_run_name in model_run_names:
    err_at_2s = lat_speed_err_collections[model_run_name][:, -1].mean()**0.5
    # print(model_run_name+' err_at_2s: ', err_at_2s)
    print(round(err_at_2s, 2))
