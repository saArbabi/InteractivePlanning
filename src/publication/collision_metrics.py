"""Use this module to check if agent plans result in a direct collision
with another vehicle.

"""
import sys
sys.path.insert(0, './src')
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from importlib import reload
from publication.scene_evolution.helpers import *
from evaluation.eval_data_obj import EvalDataObj

from src.evaluation import eval_obj
reload(eval_obj)
from src.evaluation.eval_obj import MCEVALMultiStep
eval_obj = MCEVALMultiStep()
from planner.state_indexs import StateIndxs
indxs = StateIndxs()

data_obj = EvalDataObj()
states_arr, targets_arr = data_obj.load_val_data()
state_scaler, action_scaler = loadScalers() # will set the scaler attributes

eval_obj.states_arr, eval_obj.targets_arr = states_arr, targets_arr
eval_obj.state_scaler, eval_obj.action_scaler = state_scaler, action_scaler
# %%
pred_collection = []
true_collection = []
episode_id = 2215
eval_obj.episode_ids = data_obj.load_test_episode_ids('all_density')

for episode_id in eval_obj.episode_ids.astype(int):
    print(episode_id)
    state_arr, target_arr = eval_obj.get_episode_arr(episode_id)
    states, conds = eval_obj.sequence_episode(state_arr, target_arr)
    np.random.seed(episode_id)
    tf.random.set_seed(episode_id) # each trace has a unique tf seed
    eval_obj.splits_n = 3
    if states.shape[0] < 3:
        pass
    else:
        test_data, true_state_snips = \
                eval_obj.rand_split_episode(state_arr, states, conds)
        for split_i in range(3):
            states = test_data[0][[split_i], :, :]
            conds = [item[[split_i], :, :] for item in test_data[1]]
            true_trace = true_state_snips[[split_i], :, :]
            true_trace = true_state_snips[[0], :, :]
            trace_history = np.repeat(\
                    true_trace[:, :20, 2:], traj_n, axis=0)

            states = np.repeat(states, traj_n, axis=0)
            conds = [np.repeat(cond, traj_n, axis=0) for cond in conds]

            _gen_actions, gmm_m = policy.cae_inference([states, conds])
            gen_actions = policy.gen_action_seq(_gen_actions, conds)
            bc_ders = policy.get_boundary_condition(trace_history)
            action_plans = policy.get_pred_vehicle_plans(gen_actions, bc_ders)

            pred_distances = policy.state_transition_function(trace_history[:, -1, :], \
                                                             action_plans, traj_n)

            true_acts = []
            for indx_act in indxs.indx_acts:
                true_acts.append(true_trace[:, 19:, indx_act[0]+2:indx_act[1]+3])
            true_distances = policy.state_transition_function(trace_history[0:1, -1, :], \
                                                             true_acts, 1)

            plan_likelihood = policy.get_plan_likelihoods(_gen_actions, gmm_m)

            plans_utility = policy.plan_evaluation_func(action_plans[0],
                                                     plan_likelihood,
                                                     pred_distances)

            best_plan_indx = np.argmax(plans_utility)
            best_plan = action_plans[0][best_plan_indx, :, :]

            state_0 = trace_history[:, -1, :]
            pred_distances = [dis[[best_plan_indx], :] for dis in pred_distances]
            pred_collection.extend(pred_distances)
            true_collection.extend(true_distances)

pred_collection = np.array(pred_collection)
true_collection = np.array(true_collection)
# %%
true_collection70 = true_collection.flatten().copy()
pred_collection70 = pred_collection.flatten().copy()
true_collection70 = true_collection70[true_collection70<70]
pred_collection70 = pred_collection70[pred_collection70<70]
true_collection70.min()
pred_collection70.min()
# %%
params = {
          'font.size' : 20,
          'font.family' : 'EB Garamond',
          }
plt.rcParams.update(params)
plt.style.use(['science','ieee'])

_ = plt.hist(pred_collection70.flatten(), bins=200, alpha=0.5, label='Agent plans')
_ = plt.hist(true_collection70.flatten(), bins=200, alpha=0.5, label='Human plans')
plt.plot([3.5, 3.5], [0, 1000], color='black', linestyle='--')
plt.plot([0.49, 0.49], [0, 1000], color='red', linestyle='-')
plt.plot([0.53, 0.53], [0, 1000], color='green', linestyle='-')

plt.ylim(0, 1000)
plt.xlim(-3, 70)
plt.xlabel('Vehicle distance (m)')
plt.ylabel('Histogram count')

plt.plot()
plt.legend(loc='upper right')
# plt.savefig("plan_histogram.png", dpi=500)

# %%
plt.plot(pred_collection[0][0].flatten())
plt.plot(true_collection[0][0].flatten())
# %%
