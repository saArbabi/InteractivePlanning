
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.planner.state_indexs import StateIndxs

np.set_printoptions(suppress=True)

model_names = ['cae_001', 'cae_002', 'cae_003', 'cae_004']
val_run_name = 'test_2'

true_collections = {}
pred_collections = {}
for model_name in model_names:
    exp_dir = './src/models/experiments/'+model_name+'/' + val_run_name

    with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
        true_collections[model_name] = np.array(pickle.load(handle))

    with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
        pred_collections[model_name] = np.array(pickle.load(handle))

true_collections[model_name].shape
true_collections['cae_001'].shape
pred_collections[model_name].shape
for model_name in model_names:
    print(pred_collections[model_name].shape)


indxs = {}
feature_names = [index_name, 'pc', 'act_long','act_lat',
                 index_name, 'dx', 'act_long', 'act_lat',
                 index_name, 'dx', 'act_long', 'act_lat',
                 index_name, 'dx', 'act_long', 'act_lat',
                 'lc_type', 'exists', 'exists', 'exists']

index = 0
for item_name in feature_names:
    indxs[item_name] = index
    index += 1

indxs = StateIndxs()

# %%
true_collections[model_name][0, :, 19:, 0+2]-pred_collections[model_name][0, :, :, 0]
pred_collections[model_name][0, 0, :, 0]
true_collections[model_name]

a = np.zeros([5, 5])
a
a[:, 3] = 1

np.append(a[:, 0], a[:, 3])
np.array([])
b = [a[:, 0], a[:, 3]]
b
np.array(b).flatten()

true_collections['cae_001'].shape

# %%

posx_true.shape
true_collections[model_name][-1,:,19:, indxs.indx_fadj[index_name]+2]
posx_true[-1, 0, :]
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
    for m in range(true_collections[model_name].shape[0]):
        scenario_err_arr.append(get_trace_err(posx_pred[m, :, :], posx_true[m, :, :]))
    return np.array(scenario_err_arr)

def get_rwse(scenario_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(scenario_err_arr, axis=0)**0.5


# %%


# %%
time_steps = np.linspace(0, 2., 21)
"""
rwse x position
"""
fig = plt.figure(figsize=(6, 4))
long_speed = fig.add_subplot(211)
fig.subplots_adjust(hspace=0.1)
# for model_name in model_names:
for model_name in model_names:
    scenario_err_arr = get_scenario_err('vel', model_name)
    error_total = get_rwse(scenario_err_arr)
    long_speed.plot(time_steps, error_total, label=model_name)
# model_names = ['h_lat_f_idm_act', 'h_lat_f_act', 'h_lat_act']

# legends = ['NIDM', 'Latent-Seq', 'Latent-Single', 'Latent-Single-o']
long_speed.set_ylabel('RWSE Long.speed ($ms^{-1}$)', labelpad=10)
# long_speed.set_xlabel('Time horizon (s)')
long_speed.minorticks_off()
# long_speed.set_ylim(0, 5)
long_speed.set_xticklabels([])

"""
rwse speed
"""
lat_speed = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.5)
# for model_name in model_names:

for model_name in model_names:
    scenario_err_arr = get_scenario_err('act_lat', model_name)
    error_total = get_rwse(scenario_err_arr)
    lat_speed.plot(time_steps, error_total, label=model_name)
error_total.shape
lat_speed.set_ylabel('RWSE Lat.speed ($ms^{-1}$)')
lat_speed.set_xlabel('Time horizon (s)')
lat_speed.minorticks_off()
# lat_speed.set_ylim(0, 2)
# lat_speed.set_yticks([0, 1, 2, 3])
lat_speed.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)
