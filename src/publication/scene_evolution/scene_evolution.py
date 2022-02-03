"""
#######################################  Action plans vis #######################################
"""
import sys
sys.path.insert(0, './src')
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
np.set_printoptions(suppress=True)
from evaluation.eval_data_obj import EvalDataObj
import tensorflow as tf
from publication.scene_evolution import helpers
reload(helpers)
from publication.scene_evolution.helpers import *


data_obj = EvalDataObj()
states_arr, targets_arr = data_obj.load_val_data()
state_scaler, action_scaler = loadScalers() # will set the scaler attributes

# %%
""" setup the scenario and predictive model """
from planner import action_policy
reload(action_policy)
from planner.action_policy import Policy

from models.core import cae
reload(cae)
from src.evaluation import eval_obj
reload(eval_obj)
from src.evaluation.eval_obj import MCEVALMultiStep
eval_obj = MCEVALMultiStep()
from planner.state_indexs import StateIndxs
indxs = StateIndxs()

eval_obj.states_arr, eval_obj.targets_arr = states_arr, targets_arr
eval_obj.state_scaler, eval_obj.action_scaler = state_scaler, action_scaler

model_name = 'cae_003'
model_type = 'CAE'
epoch = 50
episode_id = 2215
traj_n = 100
snap_interval = 10 # number of steps between each snapshot of a trajectory
snap_count = 3 # total number of snapshots of a given trajectory
tf_seed = 2

policy = eval_obj.load_policy(model_name, model_type, epoch)
policy.traj_n = traj_n
tf.random.set_seed(tf_seed) # each trace has a unique tf seed

state_arr, target_arr = eval_obj.get_episode_arr(episode_id)
states, conds = eval_obj.sequence_episode(state_arr, target_arr)

arrs = [state_arr, states, conds]
specs = [snap_interval, snap_count]
test_data, true_state_snips = split_episode(arrs, specs)

""" generate action plans """
pred_plans = [] # evental shape: [m scenarios, n traces, time_steps_n, states_n]
true_plans = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]
best_plan_indxs = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]

for snap_i in range(snap_count):
    states_i = test_data[0][[snap_i], :, :]
    conds_i = [item[[snap_i], :, :] for item in test_data[1]]
    true_trace = true_state_snips[[snap_i], :, :]
    trace_history = np.repeat(\
            true_trace[:, :eval_obj.obs_n, 2:], traj_n, axis=0)

    states_i = np.repeat(states_i, traj_n, axis=0)
    conds_i = [np.repeat(cond, traj_n, axis=0) for cond in conds_i]

    _gen_actions, gmm_m = policy.cae_inference([states_i, conds_i])

    gen_actions = policy.gen_action_seq(\
                        _gen_actions, conds_i)

    bc_ders = policy.get_boundary_condition(trace_history)
    action_plans = policy.get_pred_vehicle_plans(gen_actions, bc_ders)
    best_plan, best_plan_indx = policy.plan_evaluation_func(action_plans[0], _gen_actions, gmm_m)

    true_plan = []
    for veh_axis in indxs.indxs:
        act_axis = np.array([veh_axis['act_long'], veh_axis['act_lat']])+2
        true_plan.append(true_trace[:, :, act_axis])

    pred_plans.append(action_plans)
    true_plans.append(true_plan)
    best_plan_indxs.append(best_plan_indx)

#  %%
""" Plan visualisation figure """
time_steps = np.linspace(0, 3.9, 40)
ts_h = time_steps[:20]
ts_f = time_steps[19:]
subplot_xcount = 5
subplot_ycount = 3
fig, axs = plt.subplots(subplot_ycount, subplot_xcount, figsize=(18,9))
fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.17, hspace=0.1)

axs[0, 0].set_ylabel('$\ddot x_e \; (ms^{-2})$', labelpad=-2)
axs[1, 0].set_ylabel('$\ddot x_e \; (ms^{-2})$', labelpad=-2)
axs[2, 0].set_ylabel('$\ddot x_e \; (ms^{-2})$', labelpad=-2)

axs[0, 1].set_ylabel('$\dot y_e \; (ms^{-1})$', labelpad=-2)
axs[1, 1].set_ylabel('$\dot y_e \; (ms^{-1})$', labelpad=-2)
axs[2, 1].set_ylabel('$\dot y_e \; (ms^{-1})$', labelpad=-2)

axs[0, 2].set_ylabel('$\ddot x_{v_1} \; (ms^{-2})$', labelpad=-2)
axs[1, 2].set_ylabel('$\ddot x_{v_1} \; (ms^{-2})$', labelpad=-2)
axs[2, 2].set_ylabel('$\ddot x_{v_1} \; (ms^{-2})$', labelpad=-2)
axs[0, 3].set_ylabel('$\ddot x_{v_2} \; (ms^{-2})$', labelpad=-2)
axs[1, 3].set_ylabel('$\ddot x_{v_2} \; (ms^{-2})$', labelpad=-2)
axs[2, 3].set_ylabel('$\ddot x_{v_2} \; (ms^{-2})$', labelpad=-2)
axs[0, 4].set_ylabel('$\ddot x_{v_3} \; (ms^{-2})$', labelpad=-2)
axs[1, 4].set_ylabel('$\ddot x_{v_3} \; (ms^{-2})$', labelpad=-2)
axs[2, 4].set_ylabel('$\ddot x_{v_3} \; (ms^{-2})$', labelpad=-2)

for subplot_xi in range(subplot_xcount):
    for subplot_yi in range(subplot_ycount):
        axs[subplot_yi, subplot_xi].set_ylim([-2.1, 2.1])
        axs[subplot_yi, subplot_xi].set_xlim([-0.2, 4])
        axs[subplot_yi, subplot_xi].set_yticks([-2, 0, 2])
        axs[subplot_yi, subplot_xi].spines['top'].set_visible(False)
        axs[subplot_yi, subplot_xi].spines['right'].set_visible(False)
        axs[subplot_yi, subplot_xi].xaxis.set_tick_params(which="both", top=False)
        axs[subplot_yi, subplot_xi].yaxis.set_tick_params(which="both", right=False)
        # axs[subplot_yi, subplot_xi].yaxis.set_tick_params(labelright='off')
        if subplot_yi < 2:
            axs[subplot_yi, subplot_xi].set_xticklabels([])
            axs[subplot_yi, subplot_xi].get_xaxis().set_visible(False)
            axs[subplot_yi, subplot_xi].spines['bottom'].set_visible(False)
        if subplot_yi == 2:
            axs[subplot_yi, subplot_xi].xaxis.set_tick_params(which="both", top=False)
            axs[subplot_yi, subplot_xi].set_xlabel('Time (s)')

# x%%
for snap_i in range(snap_count):
    best_plan_indx = best_plan_indxs[snap_i] # index of chosen plan

    for veh_axis in range(4):
        veh_pred_plans = pred_plans[snap_i][veh_axis]
        veh_true_plans = true_plans[snap_i][veh_axis]
        if veh_axis == 0:
            # ego car
            for act_axis in range(2):
                plot_plan_space(ts_f, veh_pred_plans[:, :, act_axis], axs[snap_i, act_axis])
                # for trace_i in range(traj_n):
                #     if trace_i != best_plan_indx:
                #         pred_plan = veh_pred_plans[trace_i, :, act_axis]
                #         axs[snap_i, act_axis].plot(ts_f, pred_plan, color='grey', linewidth=0.9)

                true_plan = veh_true_plans[0, :, act_axis]
                axs[snap_i, act_axis].plot(ts_f, true_plan[19:], color='red', linestyle='--', linewidth=2.5)
                axs[snap_i, act_axis].plot(ts_h, true_plan[:20], color='black', linestyle='--', linewidth=2.5)

                pred_plan = veh_pred_plans[best_plan_indx, :, act_axis]
                axs[snap_i, act_axis].plot(ts_f, pred_plan, color='green', linewidth=2.5)
        else:
            # only plot long.acc
            for trace_i in range(50):
                pred_plan = veh_pred_plans[trace_i, :, 0]
                axs[snap_i, veh_axis+1].plot(ts_f, pred_plan, color='grey',
                                                        linewidth=0.9, alpha=0.6, linestyle='-')
            true_plan = veh_true_plans[0, :, 0]
            axs[snap_i, veh_axis+1].plot(ts_f, true_plan[19:], color='red', linestyle='--', linewidth=2.5)
            axs[snap_i, veh_axis+1].plot(ts_h, true_plan[:20], color='black', linestyle='--', linewidth=2.5)
plt.savefig("plans.png", dpi=500)

# %%
"""
#######################################  Scenario intro #######################################
"""
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

from publication.scene_evolution import viewer
reload(viewer)
from publication.scene_evolution.viewer import Viewer

plot_viewer = Viewer(env.trace_log)
plot_viewer.set_up_traffic_intro_fig()
plot_viewer.draw_speeds(state_arr[:, 2:])
plt.savefig("speeds.png", dpi=500)
# %%
"""
#######################################  Trajectory vis #######################################
"""
from planner import action_policy
reload(action_policy)
from planner.action_policy import Policy


from publication.scene_evolution import vehicles
reload(vehicles)

from publication.scene_evolution import env
reload(env)
from publication.scene_evolution.env import Env
model_name = 'cae_014'
model_type = 'CAE'
epoch = 30
policy = eval_obj.load_policy(model_name, model_type, epoch)
env = Env(state_arr[:, 2:])
env.caeveh.policy = policy
policy.traj_n = 100
tf.random.set_seed(1) # each trace has a unique tf seed

for i in range(18):
    env.step()
# env.vehicles[0].actions
# %%
for state in ['glob_y', 'speed', 'act_long', 'act_lat', 'lane_y']:
    plt.figure()
    plt.plot(env.trace_log['caeveh'][state], color='grey')
    plt.plot(env.trace_log['mveh'][state], color='red')
    plt.title(state)
    plt.grid()
# %%
from publication.scene_evolution import viewer
reload(viewer)
from publication.scene_evolution.viewer import Viewer

plot_viewer = Viewer(env.trace_log)
plot_viewer.set_up_traffic_fig()
plot_viewer.set_up_profile_fig()
plot_viewer.draw_plots()
plot_viewer.traffic_fig.savefig("traffic_fig.png", dpi=500)
plot_viewer.state_profile_fig.savefig("state_profile_fig.png", dpi=500)
