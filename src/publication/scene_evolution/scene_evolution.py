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

from matplotlib.lines import Line2D

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

model_name = 'cae_018'
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

    plan_likelihood = policy.get_plan_likelihoods(_gen_actions, gmm_m)
    abs_distances = policy.state_transition_function(\
                                            trace_history[:, -1, :],\
                                             action_plans,
                                             policy.traj_n)
    plans_utility = policy.plan_evaluation_func(action_plans[0],
                                             plan_likelihood,
                                             abs_distances)

    best_plan_indx = np.argmax(plans_utility)
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

#  %%
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

fig_x = 22
fig_y = 3
fig_instant_1, axs_instant_1 = plt.subplots(1, 5, figsize=(fig_x, fig_y))
fig_instant_2, axs_instant_2 = plt.subplots(1, 5, figsize=(fig_x, fig_y))
fig_instant_3, axs_instant_3 = plt.subplots(1, 5, figsize=(fig_x, fig_y))
figs = [fig_instant_1, fig_instant_2, fig_instant_3]
axs = [axs_instant_1, axs_instant_2, axs_instant_3]

for fig in figs:
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.)

for ax in axs:
    ax[0].set_ylabel(r'$\mathdefault{\ddot x_e \; (m/s^2)}$', labelpad=-10)
    ax[1].set_ylabel(r'$\mathdefault{\dot y_e \; (m/s)}$', labelpad=-10)
    ax[2].set_ylabel(r'$\mathdefault{\ddot x_{v_1} \; (m/s^2)}$', labelpad=-10)
    ax[3].set_ylabel(r'$\mathdefault{\ddot x_{v_2} \; (m/s^2)}$', labelpad=-10)
    ax[4].set_ylabel(r'$\mathdefault{\ddot x_{v_3} \; (m/s^2)}$', labelpad=-10)
    for ax_i in ax:
        ax_i.set_xlabel('Time (s)')
        ax_i.set_ylim([-2.1, 2.1])
        ax_i.set_xlim([0, 4])
        ax_i.set_yticks([-2, 0, 2])
        ax_i.plot([1.9, 1.9], [-3, 3], color='blue')

custom_lines = [Line2D([0], [0], color='black', lw=4, linestyle='--'),
                Line2D([0], [0], color='grey', lw=4),
                Line2D([0], [0], color='red', lw=4, linestyle='--'),
                Line2D([0], [0], color='green', lw=4),
                ]

fig_instant_1.legend(custom_lines, ['Action history', 'Anticipated human plan',\
                              'True human plan', 'Agent chosen plan'],
                  loc='upper center', bbox_to_anchor=(0.5, 1.15), edgecolor='black', ncol=4)


for snap_i in range(snap_count):
    best_plan_indx = best_plan_indxs[snap_i] # index of chosen plan

    for veh_axis in range(4):
        veh_pred_plans = pred_plans[snap_i][veh_axis]
        veh_true_plans = true_plans[snap_i][veh_axis]
        if veh_axis == 0:
            # ego car
            for act_axis in range(2):
                plot_plan_space(ts_f, veh_pred_plans[:, :, act_axis], axs[snap_i][act_axis])
                true_plan = veh_true_plans[0, :, act_axis]
                axs[snap_i][act_axis].plot(ts_f, true_plan[19:], color='red', linestyle='--', linewidth=2.5)
                axs[snap_i][act_axis].plot(ts_h, true_plan[:20], color='black', linestyle='--', linewidth=2.5)

                pred_plan = veh_pred_plans[best_plan_indx, :, act_axis]
                axs[snap_i][act_axis].plot(ts_f, pred_plan, color='green', linewidth=2.5, label='Agent plan')
        else:
            # only plot long.acc
            for trace_i in range(50):

                pred_plan = veh_pred_plans[trace_i, :, 0]
                if pred_plan.max() < 2:
                    axs[snap_i][veh_axis+1].plot(ts_f, pred_plan, color='grey',
                                                        linewidth=0.9, alpha=0.6, linestyle='-')
            true_plan = veh_true_plans[0, :, 0]
            axs[snap_i][veh_axis+1].plot(ts_f, true_plan[19:], color='red', linestyle='--', linewidth=2.5)
            axs[snap_i][veh_axis+1].plot(ts_h, true_plan[:20], color='black', linestyle='--', linewidth=2.5)

for i, fig in enumerate(figs):
    fig_name = 'plans_instant_'+str(i+1)+'.pdf'
    fig.savefig(fig_name, dpi=500, bbox_inches='tight')
#  %%

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
model_name = 'cae_018'
model_type = 'CAE'
epoch = 50
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
params = {
          'font.family': "Times New Roman",
          'legend.fontsize': 18,
          'legend.handlelength': 2}
plt.rcParams.update(params)
MEDIUM_SIZE = 18
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels


from publication.scene_evolution import viewer
reload(viewer)
from publication.scene_evolution.viewer import Viewer

plot_viewer = Viewer(env.trace_log)
plot_viewer.set_up_traffic_fig()
plot_viewer.set_up_profile_fig()
plot_viewer.draw_plots()

hspace = 0.3
plot_viewer.traffic_fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.45)
plot_viewer.state_profile_fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.45)

plot_viewer.speed_ax.legend(['Human', 'Agent'],
                      ncol=2, edgecolor='black', loc='upper right', facecolor='white')
plot_viewer.scene_t1_ax.legend(['Human', 'Agent'],
                      ncol=2, edgecolor='black', loc='upper right', facecolor='white')

# plot_viewer.traffic_fig.savefig("traffic_fig.svg", dpi=1000, bbox_inches='tight')
# plot_viewer.state_profile_fig.savefig("state_profile_fig.pdf", dpi=1000, bbox_inches='tight')
# %%
"""
#######################################  Scenario intro #######################################
"""
from publication.scene_evolution import viewer
reload(viewer)
from publication.scene_evolution.viewer import Viewer

plot_viewer = Viewer(env.trace_log)
plot_viewer.set_up_traffic_intro_fig()
plot_viewer.draw_speeds(state_arr[:, 2:])
plt.savefig("speeds.pdf", dpi=500, bbox_inches='tight')
