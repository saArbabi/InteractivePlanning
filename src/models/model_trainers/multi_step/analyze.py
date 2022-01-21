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
eval_obj.traces_n = 20
eval_obj.splits_n = 3
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
