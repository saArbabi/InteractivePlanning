import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.planner.state_indexs import StateIndxs
indxs = StateIndxs()

np.set_printoptions(suppress=True)
# %%
"""Compare different models
"""
model_names = ['cae_001', 'cae_002', 'cae_003', 'cae_004']
config_name = 'study_step_size'
val_run_name = config_name

true_collections = {}
pred_collections = {}
for model_name in model_names:
    exp_dir = './src/models/experiments/'+model_name+'/' + val_run_name

    with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
        true_collections[model_name] = np.array(pickle.load(handle))

    with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
        pred_collections[model_name] = np.array(pickle.load(handle))

true_collections[model_name].shape
pred_collections[model_name].shape

# %%
"""Compare different variants of the same model
"""
model_names = ['epoch_20',  'epoch_30', 'epoch_50']
model_name = 'cae_004'
true_collections = {}
pred_collections = {}

for val_run_name in model_names:
    exp_dir = './src/models/experiments/'+model_name+'/' + val_run_name

    with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
        true_collections[val_run_name] = np.array(pickle.load(handle))

    with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
        pred_collections[val_run_name] = np.array(pickle.load(handle))

true_collections[val_run_name].shape
pred_collections[val_run_name].shape

# %%
"""
Visualisation of model predictions. Use this for debugging.
"""
model_name = 'cae_004'
model_name = 'epoch_30'
epoch = 20

true_collection = true_collections[model_name]
pred_collection = pred_collections[model_name]

indx_acts = indxs.indx_acts
traces_n = 10
time_steps = np.linspace(0, 3.9, 40)
veh_names = ['veh_m', 'veh_y', 'veh_f', 'veh_fadj']
# scene_samples = range(2)
scene_samples = [20]
for scene_sample in scene_samples:
    fig, axs = plt.subplots(figsize=(10, 1))
    start_time = true_collection[scene_sample, 0, 0, 1]
    plt.text(0.1, 0.1,
             'model_name: '+model_name+'\n'
             'start_time: '+str(start_time)+'\n'
             'scene_sample: '+str(scene_sample)+'\n'
             'epoch: '+str(epoch), fontsize=10)
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    for veh_axis in range(4):
        for act_axis in range(2):
            true_trace = true_collection[scene_sample, 0, :, indx_acts[veh_axis][act_axis]+2]
            axs[veh_axis, act_axis].plot(time_steps[19:], true_trace[19:], color='red', linestyle='--')
            axs[veh_axis, act_axis].plot(time_steps[:20], true_trace[:20], color='black')
            for trace_axis in range(traces_n):
                pred_trace = pred_collection[scene_sample, trace_axis, :, indx_acts[veh_axis][act_axis]]
                axs[veh_axis, act_axis].plot(time_steps[19:], pred_trace,  color='grey')
                # axs[veh_axis, act_axis].scatter(time_steps[19:][::5], pred_trace[::5],  color='black')
# plt.savefig("test.png", dpi=500)
# %%
long_speed_err_collections['cae_003'].shape
errs = long_speed_err_collections['cae_004'][:, -1]
max_err = errs.max()
np.where(errs == max_err)

pred_collection[scene_sample, :, :, 0]
pred_collection[132, :, :, indx_acts[0][0]]
pred_collection[132, :, :, indx_acts[0][0]]
# %%
def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_scenario_err(index_name, model_name):
    """
    Input shpae [m scenarios, n traces, steps_n, state_index]
    Return shape [m, steps_n]
    """
    posx_true = true_collections[model_name][:,:,19:, indxs.indx_m[index_name]+2]
    posx_pred = pred_collections[model_name][:,:,:, indxs.indx_m[index_name]]
    for indx_ in [indxs.indx_y, indxs.indx_f, indxs.indx_fadj]:
        posx_true = np.append(posx_true, \
                true_collections[model_name][:,:,19:, indx_[index_name]+2], axis=0)
        posx_pred = np.append(posx_pred, \
                pred_collections[model_name][:,:,:, indx_[index_name]], axis=0)

    scenario_err_arr = []
    for m in range(posx_true.shape[0]):
        scenario_err_arr.append(get_trace_err(posx_pred[m, :, :], posx_true[m, :, :]))
    return np.array(scenario_err_arr)

def get_rwse(scenario_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(scenario_err_arr, axis=0)**0.5

# %%
time_steps = np.linspace(0, 2., 21)
"""
rwse long_speed
"""
fig = plt.figure(figsize=(6, 4))
long_speed = fig.add_subplot(211)
fig.subplots_adjust(hspace=0.1)
# for model_name in model_names:

long_speed_err_collections = {}
for model_name in model_names:
    long_speed_err_collections[model_name] = get_scenario_err('vel', model_name)
for model_name in model_names:
    scenario_err_arr = long_speed_err_collections[model_name]
    error_total = get_rwse(scenario_err_arr)
    long_speed.plot(time_steps, error_total, label=model_name)
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']

# legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
long_speed.set_ylabel('RWSE Long.speed ($ms^{-1}$)', labelpad=10)
# long_speed.set_xlabel('Time horizon (s)')
long_speed.minorticks_off()
# long_speed.set_ylim(0, 5)
long_speed.set_xticklabels([])
# d%%

"""
rwse lat_speed
"""
lat_speed = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.5)
# for model_name in model_names:

lat_speed_err_collections = {}
for model_name in model_names:
    lat_speed_err_collections[model_name] = get_scenario_err('act_lat', model_name)

for model_name in model_names:
    scenario_err_arr = lat_speed_err_collections[model_name]
    error_total = get_rwse(scenario_err_arr)
    lat_speed.plot(time_steps, error_total, label=model_name)
error_total.shape
lat_speed.set_ylabel('RWSE Lat.speed ($ms^{-1}$)')
lat_speed.set_xlabel('Time horizon (s)')
lat_speed.minorticks_off()
# lat_speed.set_ylim(0, 2)
# lat_speed.set_yticks([0, 1, 2, 3])
lat_speed.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)
