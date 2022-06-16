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

def add_to_plot(time_steps, error_total, ax, color, style):
    ax.plot(time_steps, error_total, label=model_run_name, color='red')


# %%
""" plot setup
"""
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 13,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


""" ####################################### compare CAE, MLP and LSTM #######################################"""
mc_run_name = ['all_density']

model_val_run_map = {
    'cae_024': mc_run_name, #
    'mlp_002': mc_run_name, #
    'lstm_002': mc_run_name, #
    }
model_legend_map = {
    'cae_024': 'RNN Encoder–Decoder', #
    'mlp_002': 'MLP', #
    'lstm_002': 'LSTM', #
    }
true_collections, pred_collections = get_data_log_collections(model_val_run_map)
model_run_names = list(true_collections.keys())
long_err_collections = {}

time_steps = np.linspace(0, 2., 21)
fig, axs = plt.subplots(1, 2, figsize=(9,3.7))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
for ax in axs:
    ax.set_xlim([0,2.1])

axs[0].set_xlabel(r'Time horizon (s)')
axs[0].set_ylabel(r'$\mathdefault{RWSE_x}$ (m)')
axs[1].set_xlabel(r'Time horizon (s)')
axs[1].set_ylabel(r'$\mathdefault{RWSE_y}$ (m)')
# 4%%
fig_specs = {
    'lstm_002': ['blue', ':'],
    'mlp_002': ['green', '-.'],
    'cae_024': ['red', '-']}

long_err_collections = {}
for model_run_name in model_run_names:
    long_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    fig_spec = fig_specs[model_run_name]
    axs[0].plot(time_steps, error_total, \
                label=model_run_name, color=fig_spec[0], linestyle=fig_spec[1])

axs[0].legend(list(model_legend_map.values()), loc='upper left', edgecolor='black')

lat_err_collections = {}
for model_run_name in model_run_names:
    lat_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    fig_spec = fig_specs[model_run_name]
    axs[1].plot(time_steps, error_total, \
                label=model_run_name, color=fig_spec[0], linestyle=fig_spec[1])

plt.savefig("rwse_baselines.pdf", dpi=500, bbox_inches='tight')
# %%

""" ####################################### compare step_sizes #######################################"""
mc_run_name = ['all_density']
model_val_run_map = {
    'cae_016': mc_run_name,  # "pred_step_n": 20, "step_size": 1
    'cae_017': mc_run_name,  # "pred_step_n": 10, "step_size": 2
    'cae_018': mc_run_name,  # "pred_step_n": 7, "step_size": 3
    'cae_019': mc_run_name  # "pred_step_n": 5, "step_size": 4
    }
model_legend_map = {
    'cae_016': '$\delta t=0.1\;s$',  # "pred_step_n": 20, "step_size": 1
    'cae_017': '$\delta t=0.2\;s$',  # "pred_step_n": 10, "step_size": 2
    'cae_018': '$\delta t=0.3\;s$',  # "pred_step_n": 7, "step_size": 3
    'cae_019': '$\delta t=0.4\;s$'  # "pred_step_n": 5, "step_size": 4
    }
true_collections, pred_collections = get_data_log_collections(model_val_run_map)
model_run_names = list(true_collections.keys())


time_steps = np.linspace(0, 2., 21)
fig, axs = plt.subplots(1, 2, figsize=(9,3.7))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
for ax in axs:
    ax.set_xlim([0,2.1])

axs[0].set_xlabel(r'Time horizon (s)')
axs[0].set_ylabel(r'$\mathdefault{RWSE_x}$ (m)')
axs[1].set_xlabel(r'Time horizon (s)')
axs[1].set_ylabel(r'$\mathdefault{RWSE_y}$ (m)')

# 4%%


fig_specs = {
    'cae_016': ['blue', ':'],
    'cae_017': ['green', '-.'],
    'cae_019': ['orange', '--'],
    'cae_018': ['red', '-']}

long_err_collections = {}
for model_run_name in model_run_names:
    long_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    fig_spec = fig_specs[model_run_name]
    axs[0].plot(time_steps, error_total, \
                label=model_run_name, color=fig_spec[0], linestyle=fig_spec[1])

axs[0].legend(list(model_legend_map.values()), loc='upper left', edgecolor='black')

lat_err_collections = {}
for model_run_name in model_run_names:
    lat_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    fig_spec = fig_specs[model_run_name]
    axs[1].plot(time_steps, error_total, \
                label=model_run_name, color=fig_spec[0], linestyle=fig_spec[1])


plt.savefig("rwse_step_size.pdf", dpi=500, bbox_inches='tight')
# %%
""" ####################################### compare seq length #######################################"""
mc_run_name = ['all_density']
model_val_run_map = {
    'cae_020': mc_run_name, # "pred_step_n": 1, "step_size": 3
    'cae_021': mc_run_name,  # "pred_step_n": 3, "step_size": 3
    'cae_022': mc_run_name, # "pred_step_n": 5, "step_size": 3
    'cae_018': mc_run_name # "pred_step_n": 7, "step_size": 3
    }
model_legend_map = {
    'cae_020': '$N=1$',
    'cae_021': '$N=3$',
    'cae_022': '$N=5$',
    'cae_018': '$N=7$'}

true_collections, pred_collections = get_data_log_collections(model_val_run_map)
model_run_names = list(true_collections.keys())


time_steps = np.linspace(0, 2., 21)
fig, axs = plt.subplots(1, 2, figsize=(9,3.7))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
for ax in axs:
    ax.set_xlim([0,2.1])

axs[0].set_xlabel(r'Time horizon (s)')
axs[0].set_ylabel(r'$\mathdefault{RWSE_x}$ (m)')

axs[1].set_xlabel(r'Time horizon (s)')
axs[1].set_ylabel(r'$\mathdefault{RWSE_y}$ (m)')

# 4%%
fig_specs = {
    'cae_020': ['blue', ':'],
    'cae_021': ['green', '-.'],
    'cae_022': ['orange', '--'],
    'cae_018': ['red', '-']}

long_err_collections = {}
for model_run_name in model_run_names:
    long_err_collections[model_run_name] = get_scenario_err('vel', model_run_name)
for model_run_name in model_run_names:
    scenario_err_arr = long_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    fig_spec = fig_specs[model_run_name]
    axs[0].plot(time_steps, error_total, \
                label=model_run_name, color=fig_spec[0], linestyle=fig_spec[1])

axs[0].legend(list(model_legend_map.values()), loc='upper left', edgecolor='black')

lat_err_collections = {}
for model_run_name in model_run_names:
    lat_err_collections[model_run_name] = get_scenario_err('act_lat', model_run_name)

for model_run_name in model_run_names:
    scenario_err_arr = lat_err_collections[model_run_name]
    error_total = get_rwse(scenario_err_arr)
    fig_spec = fig_specs[model_run_name]
    axs[1].plot(time_steps, error_total, \
                label=model_run_name, color=fig_spec[0], linestyle=fig_spec[1])

plt.savefig("rwse_sequence_length.pdf", dpi=500, bbox_inches='tight')
