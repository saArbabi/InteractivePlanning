import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.planner.state_indexs import StateIndxs
indxs = StateIndxs()

np.set_printoptions(suppress=True)
# %%
""" effect of step_size """
val_run_name = ['all_density']
model_val_run_map = {
    'cae_001': val_run_name,  # "pred_step_n": 20, "step_size": 1
    'cae_002': val_run_name,  # "pred_step_n": 10, "step_size": 2
    'cae_003': val_run_name,  # "pred_step_n": 7, "step_size": 3
    'cae_004': val_run_name  # "pred_step_n": 5, "step_size": 4
    }

# %%
""" effect of sequence length """
val_run_name = ['all_density']
model_val_run_map = {
    'cae_007': val_run_name, # "pred_step_n": 1, "step_size": 3
    'cae_005': val_run_name,  # "pred_step_n": 3, "step_size": 3
    'cae_006': val_run_name, # "pred_step_n": 5, "step_size": 3
    'cae_003': val_run_name, # "pred_step_n": 7, "step_size": 3
    }

# %%
""" compare CAE, MLP and LSTM """
val_run_name = ['all_density']

model_val_run_map = {
    'cae_003': val_run_name, #
    'mlp_001': val_run_name, #
    'lstm_001': val_run_name, #
    }
# %%
"""  """
val_run_name = ['all_density']

model_val_run_map = {
    'cae_003': val_run_name, #
    # 'cae_014': val_run_name, #
    'cae_010': val_run_name, #
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
"""
Visualisation of model predictions. Use this for debugging.
"""
model_run_name = 'cae_010_all_density'
# model_run_name = 'cae_003_all_density'
true_collection = true_collections[model_run_name]
pred_collection = pred_collections[model_run_name]

indx_acts = indxs.indx_acts
traces_n = 10
time_steps = np.linspace(0, 3.9, 40)
veh_names = ['veh_m', 'veh_y', 'veh_f', 'veh_fadj']

# scene_samples = range(3)
scene_samples = [100, 150, 200, 250]
for scene_sample in scene_samples:
    fig, axs = plt.subplots(figsize=(10, 1))
    start_time = true_collection[scene_sample, 0, 0, 1]
    true_collection[scene_sample, 0, 0, :]
    plt.text(0.1, 0.1,
             'model_run_name: '+model_run_name+'\n'
             'start_time: '+str(start_time)+'\n'
             'scene_sample: '+str(scene_sample), fontsize=10)
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    for v, veh_axis in enumerate([indxs.indx_m, indxs.indx_y, indxs.indx_f, indxs.indx_fadj]):
    # for v, veh_axis in enumerate([indxs.indx_m]):
        for s, state_axis in enumerate(['act_long', 'act_lat']):
            true_trace = true_collection[scene_sample, 0, :, veh_axis[state_axis]+2]
            axs[v, s].plot(time_steps[19:], true_trace[19:], color='red', linestyle='--')
            axs[v, s].plot(time_steps[:20], true_trace[:20], color='black')
            for trace_axis in range(traces_n):
                pred_trace = pred_collection[scene_sample, trace_axis, :, veh_axis[state_axis]]
                axs[v, s].plot(time_steps[19:], pred_trace,  color='grey')
                # axs[veh_axis, state_axis].scatter(time_steps[19:][::5], pred_trace[::5],  color='black')
# plt.savefig("test.png", dpi=500)

# %%
def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_scenario_err(index_name, model_run_name):
    """
    Input shpae [m scenarios, n traces, steps_n, state_index]
    Return shape [m, steps_n]
    """
    posx_true = true_collections[model_run_name][:,:,19:, indxs.indx_m[index_name]+2]
    posx_pred = pred_collections[model_run_name][:,:,:, indxs.indx_m[index_name]]
    # if index_name == 'act_lat':
    #     y_pred = np.zeros([300, 50, 21])
    #     y_true = np.zeros([300, 1, 21])
    #     for i in range(1, 21):
    #         y_true[:, :, i] = y_true[:, :, i-1] + posx_true[:, :, i-1]*0.1
    #         y_pred[:, :, i] = y_pred[:, :, i-1] + posx_pred[:, :, i-1]*0.1
    #
    #     posx_true = y_true
    #     posx_pred = y_pred
    # for indx_ in [indxs.indx_y, indxs.indx_f, indxs.indx_fadj]:
    #     posx_true = np.append(posx_true, \
    #             true_collections[model_run_name][:,:,19:, indx_[index_name]+2], axis=0)
    #     posx_pred = np.append(posx_pred, \
    #             pred_collections[model_run_name][:,:,:, indx_[index_name]], axis=0)

    scenario_err_arr = []
    for m in range(posx_true.shape[0]):
    # for m in range(30):
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

long_speed_err_collections = {}
for model_run_name in model_run_names:
    long_speed_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_speed_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    long_speed.plot(time_steps, error_total, label=model_run_name)
# model_run_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']
    print(error_total.max())
# legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
long_speed.set_ylabel('RWSE Long.speed ($ms^{-1}$)', labelpad=10)
# long_speed.set_xlabel('Time horizon (s)')
long_speed.minorticks_off()
# long_speed.set_ylim(0, 5)
long_speed.set_xticklabels([])
# %%

"""
rwse lat_speed
"""
lat_speed = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.5)

lat_speed_err_collections = {}
for model_run_name in model_run_names:
    lat_speed_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_speed_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    lat_speed.plot(time_steps, error_total, label=model_run_name)
error_total.shape
lat_speed.set_ylabel('RWSE Lat.speed ($ms^{-1}$)')
lat_speed.set_xlabel('Time horizon (s)')
lat_speed.minorticks_off()
# lat_speed.set_ylim(0, 2)
# lat_speed.set_yticks([0, 1, 2, 3])
lat_speed.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)



# %%
# lat_speed_err_collections['cae_003_high_density'].shape
# np.where(lat_speed_err_collections['cae_003_high_density']==2.8769739826493037)
# plt.plot(lat_speed_err_collections['cae_003_high_density'][22, :])
