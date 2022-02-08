import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.planner.state_indexs import StateIndxs
indxs = StateIndxs()

np.set_printoptions(suppress=True)

# %%
def get_data_log_collections(model_val_run_map):
    true_collections = {}
    pred_collections = {}
    for model_name in list(model_val_run_map.keys()):
        mc_run_names = model_val_run_map[model_name]
        for mc_run_name in mc_run_names:
            model_run_name = model_name
            exp_dir = './src/models/experiments/'+model_name+'/' + mc_run_name

            with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
                true_collections[model_run_name] = np.array(pickle.load(handle))

            with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
                pred_collections[model_run_name] = np.array(pickle.load(handle))
    return true_collections, pred_collections

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
    _pred = np.zeros([posx_true.shape[0], 50, 21])
    _true = np.zeros([posx_true.shape[0], 1, 21])
    for i in range(1, 21):
        _true[:, :, i] = _true[:, :, i-1] + posx_true[:, :, i-1]*0.1
        _pred[:, :, i] = _pred[:, :, i-1] + posx_pred[:, :, i-1]*0.1

    posx_true = _true
    posx_pred = _pred

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
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {
          'font.size' : 20,
          'font.family' : 'EB Garamond',
          }
plt.rcParams.update(params)
plt.style.use(['science','ieee'])
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

# %%

""" ####################################### compare CAE, MLP and LSTM #######################################"""

mc_run_name = ['all_density']

model_val_run_map = {
    'cae_014': mc_run_name, #
    'mlp_001': mc_run_name, #
    'lstm_001': mc_run_name, #
    }
model_legend_map = {
    'cae_014': 'RNN Encoder–Decoder', #
    'mlp_001': 'MLP', #
    'lstm_001': 'LSTM', #
    }
true_collections, pred_collections = get_data_log_collections(model_val_run_map)
model_run_names = list(true_collections.keys())


time_steps = np.linspace(0, 2., 21)
fig, axs = plt.subplots(1, 2, figsize=(9,3.5))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
for ax in axs:
    ax.set_xlim([0,2.1])

max_val = 1.1
axs[0].set_xlabel('Time horizon (s)')
axs[0].set_ylabel('$RWSE_x \; (m)$')
axs[0].yaxis.set_ticks(np.arange(0, 1.6, 0.5))
axs[0].set_ylim([-0.05, max_val])

max_val = 1.6
axs[1].set_xlabel('Time horizon (s)')
axs[1].set_ylabel('$RWSE_y \; (m)$')
axs[1].yaxis.set_ticks(np.arange(0, 1.6, 0.5))
axs[1].set_ylim([-0.05, max_val])

# 4%%
long_err_collections = {}
for model_run_name in model_run_names:
    long_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    if model_run_name == 'cae_014':
        axs[0].plot(time_steps, error_total, label=model_run_name, color='red')
    elif model_run_name == 'mlp_001':
        axs[0].plot(time_steps, error_total, label=model_run_name, color='black')
    else:
        axs[0].plot(time_steps, error_total, label=model_run_name)

axs[0].legend(list(model_legend_map.values()), loc='upper left')

lat_err_collections = {}
for model_run_name in model_run_names:
    lat_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    if model_run_name == 'cae_014':
        axs[1].plot(time_steps, error_total, label=model_run_name, color='red')
    elif model_run_name == 'mlp_001':
        axs[1].plot(time_steps, error_total, label=model_run_name, color='black')
    else:
        axs[1].plot(time_steps, error_total, label=model_run_name)

plt.savefig("rwse_models.png", dpi=500)
# %%

""" ####################################### compare step_sizes #######################################"""

mc_run_name = ['all_density']

mc_run_name = ['all_density']
model_val_run_map = {
    'cae_001': mc_run_name,  # "pred_step_n": 20, "step_size": 1
    'cae_002': mc_run_name,  # "pred_step_n": 10, "step_size": 2
    'cae_003': mc_run_name,  # "pred_step_n": 7, "step_size": 3
    'cae_004': mc_run_name  # "pred_step_n": 5, "step_size": 4
    }
model_legend_map = {
    'cae_001': '$\delta t=0.1\;s$',  # "pred_step_n": 20, "step_size": 1
    'cae_002': '$\delta t=0.2\;s$',  # "pred_step_n": 10, "step_size": 2
    'cae_003': '$\delta t=0.3\;s$',  # "pred_step_n": 7, "step_size": 3
    'cae_004': '$\delta t=0.4\;s$'  # "pred_step_n": 5, "step_size": 4
    }
true_collections, pred_collections = get_data_log_collections(model_val_run_map)
model_run_names = list(true_collections.keys())


time_steps = np.linspace(0, 2., 21)
fig, axs = plt.subplots(1, 2, figsize=(9,3.5))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
for ax in axs:
    ax.set_xlim([0,2.1])

max_val = 2.2
axs[0].set_xlabel('Time horizon (s)')
axs[0].set_ylabel('$RWSE_x \; (m)$')
axs[0].yaxis.set_ticks(np.arange(0, max_val, 0.5))
axs[0].set_ylim([-0.05, max_val])

max_val = 3.2
axs[1].set_xlabel('Time horizon (s)')
axs[1].set_ylabel('$RWSE_y \; (m)$')
axs[1].yaxis.set_ticks(np.arange(0, max_val, 1))
axs[1].set_ylim([-0.05, max_val])

# 4%%
long_err_collections = {}
for model_run_name in model_run_names:
    long_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    if model_run_name == 'cae_004':
        axs[0].plot(time_steps, error_total, label=model_run_name, color='red')
    elif model_run_name == 'cae_002':
        axs[0].plot(time_steps, error_total, label=model_run_name, color='green')
    else:
        axs[0].plot(time_steps, error_total, label=model_run_name)

axs[0].legend(list(model_legend_map.values()), loc='upper left')

lat_err_collections = {}
for model_run_name in model_run_names:
    lat_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    if model_run_name == 'cae_004':
        axs[1].plot(time_steps, error_total, label=model_run_name, color='red')
    elif model_run_name == 'cae_002':
        axs[1].plot(time_steps, error_total, label=model_run_name, color='green')
    else:
        axs[1].plot(time_steps, error_total, label=model_run_name)

plt.savefig("rwse_step_size.png", dpi=500)
# %%
""" ####################################### compare step_sizes #######################################"""
mc_run_name = ['all_density']
model_val_run_map = {
    'cae_007': mc_run_name, # "pred_step_n": 1, "step_size": 3
    'cae_005': mc_run_name,  # "pred_step_n": 3, "step_size": 3
    'cae_006': mc_run_name, # "pred_step_n": 5, "step_size": 3
    'cae_003': mc_run_name, # "pred_step_n": 7, "step_size": 3
    }
model_legend_map = {
    'cae_007': '$N=1$',
    'cae_005': '$N=3$',
    'cae_006': '$N=5$',
    'cae_003': '$N=7$'
    }

true_collections, pred_collections = get_data_log_collections(model_val_run_map)
model_run_names = list(true_collections.keys())


time_steps = np.linspace(0, 2., 21)
fig, axs = plt.subplots(1, 2, figsize=(9,3.5))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
for ax in axs:
    ax.set_xlim([0,2.1])

max_val = 2.2
axs[0].set_xlabel('Time horizon (s)')
axs[0].set_ylabel('$RWSE_x \; (m)$')
axs[0].yaxis.set_ticks(np.arange(0, max_val, 0.5))
axs[0].set_ylim([-0.05, max_val])

max_val = 3.2
axs[1].set_xlabel('Time horizon (s)')
axs[1].set_ylabel('$RWSE_y \; (m)$')
axs[1].yaxis.set_ticks(np.arange(0, max_val, 1))
axs[1].set_ylim([-0.05, max_val])

# 4%%
long_err_collections = {}
for model_run_name in model_run_names:
    long_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    if model_run_name == 'cae_003':
        axs[0].plot(time_steps, error_total, label=model_run_name, color='red')
    elif model_run_name == 'cae_005':
        axs[0].plot(time_steps, error_total, label=model_run_name, color='green')
    else:
        axs[0].plot(time_steps, error_total, label=model_run_name)

axs[0].legend(list(model_legend_map.values()), loc='upper left')

lat_err_collections = {}
for model_run_name in model_run_names:
    lat_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    if model_run_name == 'cae_003':
        axs[1].plot(time_steps, error_total, label=model_run_name, color='red')
    elif model_run_name == 'cae_005':
        axs[1].plot(time_steps, error_total, label=model_run_name, color='green')
    else:
        axs[1].plot(time_steps, error_total, label=model_run_name)

# plt.savefig("rwse_sequence_length.png", dpi=500)
