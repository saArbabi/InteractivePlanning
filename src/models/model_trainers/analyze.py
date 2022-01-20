import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
np.set_printoptions(suppress=True)
import os
import pickle
import sys
import json
from importlib import reload
from src.evaluation.eval_data_obj import EvalDataObj

reload(pyplot)
sys.path.insert(0, './src')
# %%
"""
Load data
"""
data_obj = EvalDataObj()
states_arr, targets_arr = data_obj.load_val_data()
# %%
"""
Load evaluation object (This is needed for prepping test data )
"""
from src.evaluation import eval_obj
reload(eval_obj)
from src.evaluation.eval_obj import MCEVAL


model_name = 'cae_003'
eval_obj = MCEVAL()
config = eval_obj.read_model_config(model_name)
eval_obj.states_arr = states_arr
eval_obj.targets_arr = targets_arr
# %%
"""
Load policy
"""
from planner import action_policy
reload(action_policy)
from src.planner.action_policy import Policy
policy = Policy()
epoch = 20
policy.load_model(config, epoch)
eval_obj.policy = policy

episode_id = 129
true_collection, pred_collection = eval_obj.run_episode(episode_id)
true_collection, pred_collection = np.array(true_collection), np.array(pred_collection)
pred_collection.shape
true_collection.shape
true_collection[:, :, 19:, :].shape
# %%



# %%
"""
Plot loss
"""
with open(exp_dir+'/'+'losses.pickle', 'rb') as handle:
    losses = pickle.load(handle)
plt.figure()
plt.plot(losses['test_mseloss'], label='test_mseloss')
plt.plot(losses['train_mseloss'], label='train_mseloss')
plt.grid()
plt.legend()
plt.figure()
plt.plot(losses['test_klloss'], label='test_klloss')
plt.plot(losses['train_klloss'], label='train_klloss')
plt.legend()
plt.grid()

# %%
"""
Compare losses
"""
losses = {}
# for name in ['neural_idm_105', 'neural_idm_106']:
for name in ['latent_mlp_09', 'latent_mlp_10']:
    with open('./src/models/experiments/'+name+'/'+'losses.pickle', 'rb') as handle:
        losses[name] = pickle.load(handle)

plt.figure()
for name, loss in losses.items():
    plt.plot(loss['test_mseloss'], label=name)
    # plt.plot(loss['train_mseloss'], label='train_mseloss')
    plt.grid()
    plt.legend()

plt.figure()
for name, loss in losses.items():
    plt.plot(loss['test_klloss'], label=name)
    # plt.plot(loss['train_mseloss'], label='train_mseloss')
    plt.grid()
    plt.legend()
# %%
"""
Find bad examples ?
"""

# %%
"""
Visualisation of model predictions. Use this for debugging.
"""
m = 0
indx_acts = eval_obj.indxs.indx_acts
obs_n = 20
traces_n = 3
time_steps = np.linspace(0, 3.9, 40)
veh_names = ['veh_m', 'veh_y', 'veh_f', 'veh_fadj']
scene_samples = [0]
scene_samples = range(1)
for scene_sample in scene_samples:
    fig, axs = plt.subplots(figsize=(10, 1))
    start_time = true_collection[scene_sample, 0, 0, 1]
    plt.text(0.1, 0.1,
             'model_name: '+model_name+'\n'
             'start_time: '+str(start_time)+'\n'
             'scene_sample: '+str(scene_sample)+'\n'
             'epoch: '+str(epoch), fontsize=10)
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    for veh_axis in range(4):
        for act_axis in range(2):
            true_trace = true_collection[scene_sample, 0, :, indx_acts[veh_axis][act_axis]+2]
            axs[veh_axis, act_axis].plot(time_steps[19:], true_trace[19:], color='red', linestyle='--')
            axs[veh_axis, act_axis].plot(time_steps[:20], true_trace[:20], color='black')
            for trace_axis in range(traces_n):
                pred_trace = pred_collection[scene_sample, trace_axis, :, indx_acts[veh_axis][act_axis]]
                axs[veh_axis, act_axis].plot(time_steps[19:], pred_trace,  color='grey')
                axs[veh_axis, act_axis].scatter(time_steps[19:][::3], pred_trace[::3],  color='black')
# plt.savefig("test.png", dpi=500)
