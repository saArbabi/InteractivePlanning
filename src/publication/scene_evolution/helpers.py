import pickle
import numpy as np

def loadScalers():
    with open('./src/datasets/'+'state_scaler', 'rb') as f:
        state_scaler = pickle.load(f)
    with open('./src/datasets/'+'action_scaler', 'rb') as f:
        action_scaler = pickle.load(f)
    return state_scaler, action_scaler

def split_episode(arrs, specs):
    """
    Inputs:
    state_arr: un-scaled state array
    states: processed (scaled + sequenced) state histories to feed model
    conds: processed (scaled + sequenced) action conditionals to feed model
    """
    state_arr, states, conds = arrs
    snap_interval, snap_count = specs
    snippets = np.arange(states.shape[0])[::snap_interval]

    states = states[snippets, :, 2:]
    conds = [cond[snippets, :, 2:] for cond in conds]

    true_state_snips = []
    for start_step in snippets:
        true_state_snips.append(state_arr[start_step:start_step + 40, :])

    test_data = [states, conds] # to feed to model
    return test_data, np.array(true_state_snips)

def plot_plan_space(ts_f, veh_pred_plans, ax):
    plan_stdev = veh_pred_plans[:, :].std(axis=0)
    plan_mean = veh_pred_plans[:, :].mean(axis=0)
    ax.fill_between(ts_f, plan_mean+plan_stdev, \
                    plan_mean-plan_stdev, color='orange', alpha=0.3)

    ax.plot(ts_f, plan_mean+plan_stdev, color='orange')
    ax.plot(ts_f, plan_mean-plan_stdev, color='orange')
