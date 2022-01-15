import pickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
np.set_printoptions(suppress=True)
from importlib import reload
import pickle
import os
import json
import sys
sys.path.insert(0, './src')
from data.preprocessing.data_obj import DataObj
print(os.getcwd())
import time


# %%


# %%
"""
Load data
"""

# %%
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "components_n": 5,
    "batch_size": 512,

},
"data_config": {"obs_n": 20,
                "pred_step_n": 10,
                "step_size": 1,
                "Note": ""
},
"exp_id": "NA",
"Note": "NA"
}

class Trainer():
    def __init__(self):
        self.train_losses = {
            'train_llloss_m': [],
            'train_llloss_y': [],
            'train_llloss_f': [],
            'train_llloss_fadj' : []}

        self.test_losses = {
            'test_llloss_m': [],
            'test_llloss_y': [],
            'test_llloss_f': [],
            'test_llloss_fadj' : []}

        self.epoch_count = 0
        self.initiate_model()

    def initiate_model(self):
        from models.core import cae
        reload(cae)
        from models.core.cae import CAE
        self.model = CAE(config, model_use='training')

    def load_pre_trained(self, epoch_count):
        exp_dir = self.exp_dir+'/model_epo'+epoch_count
        self.epoch_count = int(epoch_count)
        self.model.load_weights(exp_dir).expect_partial()

        with open(os.path.dirname(exp_dir)+'/'+'train_losses.pickle', 'rb') as handle:
            self.train_losses = pickle.load(handle)

        with open(os.path.dirname(exp_dir)+'/'+'test_losses.pickle', 'rb') as handle:
            self.test_losses = pickle.load(handle)

    def update_config(self):
        config['train_info'] = {}
        config['train_info']['epoch_count'] = self.epoch_count

        with open(self.exp_dir+'/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def train(self, train_input, val_input, epochs):
        for epoch in range(epochs):
            t0 = time.time()

            self.epoch_count += 1
            self.model.train_loop(train_input)
            self.model.test_loop(val_input)

            self.train_losses['train_llloss_m'].append(\
                    round(self.model.train_llloss_m.result().numpy().item(), 2))
            self.train_losses['train_llloss_y'].append(\
                    round(self.model.train_llloss_y.result().numpy().item(), 2))
            self.train_losses['train_llloss_f'].append(\
                    round(self.model.train_llloss_f.result().numpy().item(), 2))
            self.train_losses['train_llloss_fadj'].append(\
                    round(self.model.train_llloss_fadj.result().numpy().item(), 2))

            self.test_losses['test_llloss_m'].append(\
                    round(self.model.test_llloss_m.result().numpy().item(), 2))
            self.test_losses['test_llloss_y'].append(\
                    round(self.model.test_llloss_y.result().numpy().item(), 2))
            self.test_losses['test_llloss_f'].append(\
                    round(self.model.test_llloss_f.result().numpy().item(), 2))
            self.test_losses['test_llloss_fadj'].append(\
                    round(self.model.test_llloss_fadj.result().numpy().item(), 2))

            t1 = time.time()

            print(self.epoch_count, 'epochs completed')
            print(t1-t0,'s to complete epoch')

    def save_model(self):
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.update_config()
        check_point_dir = self.exp_dir+'/model_epo{epoch}'.format(\
                                                    epoch=self.epoch_count)
        if not os.path.exists(check_point_dir+'.index'):
            self.model.save_weights(check_point_dir)
        else:
            print('This checkpoint is already saved')

    def save_loss(self):
        with open(self.exp_dir+'/train_losses.pickle', 'wb') as handle:
            pickle.dump(self.train_losses, handle)

        with open(self.exp_dir+'/test_losses.pickle', 'wb') as handle:
            pickle.dump(self.test_losses, handle)

# data_objs = DataObj(config).loadData()
# train_input, val_input = data_objs[0:3], data_objs[3:]
# val_input[1][1][0].shape

tf.random.set_seed(2021)
model_trainer = Trainer()
exp_id = 'cae_'+'002'
model_trainer.exp_dir = './src/models/experiments/'+exp_id
config['exp_id'] = exp_id
# model_trainer.train(train_input, val_input, epochs=1)
# model_trainer.load_pre_trained(epoch_count='5')
# %%
################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(train_input, val_input, epochs=5)
################## ##### ##################
################## ##### ##################
################## ##### ##################

# %%
"""
Plot losses
"""
epochs = range(1, model_trainer.epoch_count+1)
for tr, te in zip(model_trainer.train_losses.keys(), model_trainer.test_losses.keys()):
    plt.figure()
    plt.plot(epochs, model_trainer.train_losses[tr], color='red', label=tr)
    plt.scatter(epochs, model_trainer.train_losses[tr], color='red')
    plt.plot(epochs, model_trainer.test_losses[te], color='blue', label=te)
    plt.scatter(epochs, model_trainer.test_losses[te], color='blue')
    plt.grid()
    plt.legend()

# %%
model_trainer.save_model()
model_trainer.save_loss()
##############
##############
##############
##############

# %%
# states_, targets_, conditions_ = data_objs[0:3] # train
states_, targets_, conditions_ = data_objs[3:] # val
# %%
state_col = ['episode_id', 'vel', 'pc', 'act_long','act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'lc_type', 'exists', 'exists', 'exists']
i = 0
indxs_m = {}
indxs = [[2, 3], [6, 7], [10, 11], [14, 15]]
# %%


# %%
from planner import action_policy
reload(action_policy)
from planner.action_policy import Policy
policy = Policy(config)
policy.load_model(epoch=10)

traj_n = 5
steps_n = 10
obs_n = 20
np.random.seed(2022)
eval_samples = np.random.randint(0, 120, 1) # samples from dataset to evaluate
# eval_samples = [112]
veh_names = ['veh_m', 'veh_y', 'veh_f', 'veh_fadj']
for eval_sample in eval_samples:
    fig, axs = plt.subplots(figsize=(10, 1))
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    state_history = states_[steps_n][[eval_sample], :, :]
    conds = [conditions_[steps_n][i][[eval_sample], :, :] for i in range(4)]
    inputs = [state_history, conds]
    gen_actions = policy.gen_action_seq(inputs, traj_n, steps_n)
    for veh_axis in range(4):
        true_acts_f = targets_[steps_n][veh_axis][[eval_sample], :, :]
        true_acts_f = np.insert(true_acts_f, 0, conds[veh_axis][:, 0, :], axis=1)
        true_acts_c = np.insert(true_acts_f, 0, conds[veh_axis][:, 0, :], axis=1)
        true_acts_p = states_[steps_n][[eval_sample], :, indxs[veh_axis]]
        for act_axis in range(2):
            axs[veh_axis, act_axis].plot(range(obs_n-1, obs_n+steps_n), \
                            true_acts_f[0, :, act_axis], 'red', label=veh_names[veh_axis])

            axs[veh_axis, act_axis].plot(range(obs_n-1, obs_n+steps_n-1), \
                            conds[veh_axis][0, :, act_axis], 'blue', label=veh_names[veh_axis])

            axs[veh_axis, act_axis].plot(range(obs_n), \
                            true_acts_p[act_axis, : ], 'black', label=veh_names[veh_axis])

            axs[veh_axis, act_axis].legend()
            for trace_axis in range(traj_n):
                axs[veh_axis, act_axis].plot(range(obs_n-1, obs_n+steps_n), \
                            gen_actions[veh_axis][trace_axis, :, act_axis], 'grey')

# %%
a = 1, 2

test[0].shape
sample = 1

plt.figure()
plt.plot(range(1, 11), gen_actions[0][sample, :, 0], 'red')
# plt.plot(conds[0][0, :, 0])
plt.plot(test[0][sample, :, 0], color='green')
plt.scatter(range(11), test[0][sample, :, 0], color='green')

plt.figure()
plt.plot(range(1, 11), gen_actions[0][sample, :, 1], 'red')
# plt.plot(conds[0][0, :, 0])
plt.plot(test[0][sample, :, 1], color='green')
plt.scatter(range(11), test[0][sample, :, 1], color='green')


# %%


steps_n = 10
np.arange(0, traj_len+0.1, 0.3)
np.arange(0, traj_len+0.1, self.step_len)

time_coarse = np.linspace(0, 0.3*(steps_n), steps_n+1)
time_fine = np.arange(0, time_coarse[-1]+0.05, 0.1)
time_coarse.shape
f = CubicSpline(time_coarse, test[0][:, :, :], axis=1)
coefs = np.stack(f.c, axis=2)
plans = f(time_fine)
plans.shape

plt.plot(time_coarse, test[0][0, :, 1], color='red')
plt.plot(time_fine, plans[0, :, 1])
# %%


train_input[1][10][0][0:1, :, :]
cond = [train_input[2][10][i][0:1, :, :] for i in range(4)]
dis = model_trainer.model.call([train_input[0][10][0:1, :, :], cond])
dis[0].sample()

# %%


targets_[steps_n][1][[eval_sample], :, :]
