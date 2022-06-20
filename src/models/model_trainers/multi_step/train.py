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
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "components_n": 5,
     "allowed_error": 0,
    "batch_size": 512,

},
"data_config": {"obs_n": 20,
                "pred_step_n": 10,
                "step_size": 1,
                "Note": "lc and lk episodes."
},
"model_name": "NA",
"Note": ""
}


data_objs = DataObj(config).loadData()
train_input, test_input = data_objs[0:3], data_objs[3:]
# plt.hist(train_input[0][2][:, 0, -4])
# train_input[0][7][0, -1, :]
# train_input[2][7][0][0, 0, :]
# train_input[1][7][0][0, 0, :]
# %%
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
        print('Make sure the data corresponding to this model is loaded')
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
        config['model_name'] = self.model_name

        with open(self.exp_dir+'/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def train(self, train_input, test_input, epochs):
        for epoch in range(epochs):

            t0 = time.time()

            self.epoch_count += 1
            self.model.train_loop(train_input)
            self.model.test_loop(test_input)

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
        self.save_model()

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


tf.random.set_seed(2021)
model_trainer = Trainer()
model_trainer.model_name = 'cae_'+'027'
model_trainer.exp_dir = './src/models/experiments/'+model_trainer.model_name
config
# model_trainer.train(train_input, test_input, epochs=1)
# model_trainer.load_pre_trained(epoch_count='50')
# %%
################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(train_input, test_input, epochs=30)
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
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

for ax in axs.flatten():
    ax.set_ylabel('Log loss')
    ax.set_xlabel('Training epoch')
    ax.grid()

axs[0, 0].plot(model_trainer.train_losses['train_llloss_m'], '-o', color='red', label='Train loss')
axs[0, 0].plot(model_trainer.test_losses['test_llloss_m'], '-o', color='blue', label='Validation loss')
axs[0, 0].legend()
axs[0, 0].set_title('Vehicle $e$ loss')

axs[0, 1].plot(model_trainer.train_losses['train_llloss_y'], '-o', color='red', label='Train loss')
axs[0, 1].plot(model_trainer.test_losses['test_llloss_y'], '-o', color='blue', label='Validation loss')
axs[0, 1].legend()
axs[0, 1].set_title('Vehicle $v_1$ loss')

axs[1, 0].plot(model_trainer.train_losses['train_llloss_f'], '-o', color='red', label='Train loss')
axs[1, 0].plot(model_trainer.test_losses['test_llloss_f'], '-o', color='blue', label='Validation loss')
axs[1, 0].legend()
axs[1, 0].set_title('Vehicle $v_2$ loss')

axs[1, 1].plot(model_trainer.train_losses['train_llloss_fadj'], '-o', color='red', label='Train loss')
axs[1, 1].plot(model_trainer.test_losses['test_llloss_fadj'], '-o', color='blue', label='Validation loss')
axs[1, 1].legend()
axs[1, 1].set_title('Vehicle $v_3$ loss')
fig.savefig('losses.pdf', dpi=500, bbox_inches='tight')



# %%
model_trainer.save_model()
model_trainer.save_loss()
##############
##############
##############
##############
