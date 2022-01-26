"""
Action plans
"""
import sys
sys.path.insert(0, './src')
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
np.set_printoptions(suppress=True)
from evaluation.eval_data_obj import EvalDataObj
import pickle



# policy = Policy()


def loadScalers():
    with open('./src/datasets/'+'state_scaler', 'rb') as f:
        state_scaler = pickle.load(f)
    with open('./src/datasets/'+'action_scaler', 'rb') as f:
        action_scaler = pickle.load(f)
    return state_scaler, action_scaler

data_obj = EvalDataObj()
states_arr, targets_arr = data_obj.load_val_data()
state_scaler, action_scaler = loadScalers() # will set the scaler attributes

# %%
from models.core import cae
reload(cae)
from src.evaluation import eval_obj
reload(eval_obj)
from src.evaluation.eval_obj import MCEVALMultiStep
from planner import action_policy
reload(action_policy)
from src.planner.action_policy import Policy
eval_obj = MCEVALMultiStep()
from planner.state_indexs import StateIndxs
indxs = StateIndxs()

eval_obj.states_arr, eval_obj.targets_arr = states_arr, targets_arr
eval_obj.state_scaler, eval_obj.action_scaler = state_scaler, action_scaler

model_name = 'cae_003'
model_type = 'CAE'
epoch = 50
episode_id = 2215
traces_n = 10
snap_interval = 10 # number of steps between each snapshot of a trajectory
snap_count = 3 # total number of snapshots of a given trajectory
np.arange(50=)[::10]
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



policy = eval_obj.load_policy(model_name, model_type, epoch)

state_arr, target_arr = eval_obj.get_episode_arr(episode_id)
states, conds = eval_obj.sequence_episode(state_arr, target_arr)

arrs = [state_arr, states, conds]
specs = [snap_interval, snap_count]
test_data, true_state_snips = split_episode(arrs, specs)

pred_plans = [] # evental shape: [m scenarios, n traces, time_steps_n, states_n]
true_plans = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]
veh_axiss = [indxs.indx_m, indxs.indx_y, indxs.indx_f, indxs.indx_fadj]

for snap_i in range(snap_count):
    states_i = test_data[0][[snap_i], :, :]
    conds_i = [item[[snap_i], :, :] for item in test_data[1]]

    true_trace = true_state_snips[[snap_i], :, :]

    trace_history = np.repeat(\
            true_trace[:, :eval_obj.obs_n, 2:], traces_n, axis=0)

    gen_actions = policy.gen_action_seq(\
                        [states_i, conds_i], traj_n=traces_n)

    bc_ders = policy.get_boundary_condition(trace_history)
    action_plans = policy.construct_policy(gen_actions, bc_ders, traces_n)

    true_plan = []
    for veh_axis in veh_axiss:
        act_axis = np.array([veh_axis['act_long'], veh_axis['act_lat']])+2
        true_plan.append(true_trace[:, :, act_axis])

    pred_plans.append(action_plans)
    true_plans.append(true_plan)

action_plans[0].shape
len(action_plans)
# %%
time_steps = np.linspace(0, 3.9, 40)
subplot_xcount = 5
subplot_ycount = 3
fig, axs = plt.subplots(subplot_ycount, subplot_xcount, figsize=(18,9))
fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.17, hspace=0.1)

for subplot_xi in range(subplot_xcount):
    for subplot_yi in range(subplot_ycount):
        axs[subplot_yi, subplot_xi].spines['top'].set_visible(False)
        axs[subplot_yi, subplot_xi].spines['right'].set_visible(False)
        axs[subplot_yi, subplot_xi].set_ylim([-2.1, 2.1])
        axs[subplot_yi, subplot_xi].set_xlim([0, 4])
        axs[subplot_yi, subplot_xi].set_yticks([-2, 0, 2])
        axs[subplot_yi, subplot_xi].set_ylabel('$x \; (ms^{-2})$', labelpad=-2)
        if subplot_yi < 2:
            axs[subplot_yi, subplot_xi].set_xticklabels([])
            axs[subplot_yi, subplot_xi].get_xaxis().set_visible(False)
            axs[subplot_yi, subplot_xi].spines['bottom'].set_visible(False)
        if subplot_yi == 2:
            axs[subplot_yi, subplot_xi].set_xlabel('Time (s)')



# 5%%
for snap_i in range(snap_count):
    for veh_axis in range(4):
        if veh_axis == 0:
            # ego car
            for act_axis in range(2):
                for trace_i in range(traces_n):
                    pred_plan = pred_plans[snap_i][veh_axis][trace_i, :, act_axis]
                    axs[snap_i, act_axis].plot(time_steps[19:], pred_plan, color='grey', linewidth=0.9)
                true_plan = true_plans[snap_i][veh_axis][0, :, act_axis]
                axs[snap_i, act_axis].plot(time_steps[19:], true_plan[19:], color='red', linestyle='--')
                axs[snap_i, act_axis].plot(time_steps[:20], true_plan[:20], color='black', linestyle='--')
        else:
            # only plot long.acc
            for trace_i in range(traces_n):
                pred_plan = pred_plans[snap_i][veh_axis][trace_i, :, 0]
                axs[snap_i, veh_axis+1].plot(time_steps[19:], pred_plan, color='grey', linewidth=0.9)
            true_plan = true_plans[snap_i][veh_axis][0, :, 0]
            axs[snap_i, veh_axis+1].plot(time_steps[19:], true_plan[19:], color='red', linestyle='--')
            axs[snap_i, veh_axis+1].plot(time_steps[:20], true_plan[:20], color='black', linestyle='--')

# %%

for snap_i in range(snap_count):
    plt.figure( )
    plt.plot(true_plans[snap_i][0][0, 19:, 0], color='red')


    for trace_i in range(traces_n):
        plt.plot(pred_plans[snap_i][0][trace_i, :, 0], color='grey')


# %%



# %%
"""
Trajectory
"""
