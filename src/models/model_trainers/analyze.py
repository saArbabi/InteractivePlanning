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


"""
Load data
"""
model_name = 'cae_003'
exp_dir = './src/models/experiments/'+model_name
with open(exp_dir+'/'+'config.json', 'rb') as handle:
    config = json.load(handle)
data_obj = EvalDataObj(config['data_config'])

# %%
"""
Load policy (with config file)
"""
from src.planner import action_policy
reload(action_policy)
from src.planner.action_policy import Policy
policy = Policy()
policy.load_model(model_name)
# %%
"""
Load evaluation object (This is needed for prepping test data )
"""
from src.evaluation import eval_obj
reload(eval_obj)
from src.evaluation.eval_obj import MCEVAL
eval_obj = MCEVAL()
eval_obj.data_obj = data_obj.data_obj
eval_obj.obs_n = eval_obj.data_obj.obs_n
eval_obj.step_size = eval_obj.data_obj.step_size
eval_obj.pred_step_n = np.ceil(21/eval_obj.step_size).astype('int')
eval_obj.states_arr = data_obj.states_arr
eval_obj.targets_arr = data_obj.targets_arr
eval_obj.policy =  policy
# %%
episode_id = 2815

true_collection, pred_collection = eval_obj.run_episode(episode_id)
true_collection, pred_collection = np.array(true_collection), np.array(pred_collection)
true_collection.shape

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

# %%
"""
Visualisation of model predictions. Use this for debugging.
"""
m = 0
indx_acts = eval_obj.indxs.indx_acts
obs_n = 20
traces_n = 5

veh_names = ['veh_m', 'veh_y', 'veh_f', 'veh_fadj']
scene_samples = [0]
scene_samples = range(6)
for scene_sample in scene_samples:
    fig, axs = plt.subplots(figsize=(10, 1))
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    for veh_axis in range(4):
        for act_axis in range(2):
            true_trace = true_collection[scene_sample, 0, :, indx_acts[veh_axis][act_axis]+2]
            axs[veh_axis, act_axis].plot(time_steps[19:], true_trace[19:], color='red', linestyle='--')
            axs[veh_axis, act_axis].plot(time_steps[:20], true_trace[:20], color='black')

            for trace_axis in range(traces_n):
                pred_trace = pred_collection[scene_sample, trace_axis, :, indx_acts[veh_axis][act_axis]]
                axs[veh_axis, act_axis].plot(time_steps[19:], pred_trace,  color='grey')
