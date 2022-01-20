
import pickle
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

model_names = ['cae_001', 'cae_002', 'cae_003']
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
pred_collections[model_name].shape
# %%
true_collections[model_name][0, :, 19:, 0+2]-pred_collections[model_name][0, :, :, 0]
pred_collections[model_name][0, 0, :, 0]
true_collections[model_name]
# %%

def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_veh_err(index, model_name):
    """
    Input shpae [m, n, steps_n, state_index]
    Return shape [m, steps_n]
    """
    posx_true = true_collections[model_name][:,:,19:, index+2] #  first two indexes are episode and time_stamps
    posx_pred = pred_collections[model_name][:,:,:, index] #

    vehs_err_arr = [] # vehicles error array
    for m in range(true_collections[model_name].shape[0]):
        vehs_err_arr.append(get_trace_err(posx_pred[m, :, :], posx_true[m, :, :]))
    return np.array(vehs_err_arr)

def get_rwse(vehs_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(vehs_err_arr, axis=0)**0.5


# %%


# %%


time_steps = np.linspace(0, 2., 21)
fig = plt.figure(figsize=(6, 4))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.1)
# for model_name in model_names:

for model_name in model_names:
    vehs_err_arr = get_veh_err(0, model_name)
    error_total = get_rwse(vehs_err_arr)
    speed_axis.plot(time_steps, error_total, label=model_name)
error_total.shape
speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)')
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
# speed_axis.set_ylim(0, 2)
# speed_axis.set_yticks([0, 1, 2, 3])
speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)
