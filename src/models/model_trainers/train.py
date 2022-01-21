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
"""
Load data
"""
config = {
 "model_config": {
     "learning_rate": 1e-3,
     "components_n": 5,
    "batch_size": 512,

},
"data_config": {"obs_n": 20,
                "pred_step_n": 7,
                "step_size": 3,
                "Note": ""
},
"model_name": "NA",
"Note": "Without guided learning"
}

from data.preprocessing import data_prep
from data.preprocessing import data_obj
reload(data_prep)
reload(data_obj)
data_objs = DataObj(config).loadData()
train_input, val_input = data_objs[0:3], data_objs[3:]
# train_input[0][7].shape
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


tf.random.set_seed(2021)
model_trainer = Trainer()
model_name = 'cae_'+'003'
model_trainer.exp_dir = './src/models/experiments/'+model_name
config['model_name'] = model_name
# model_trainer.train(train_input, val_input, epochs=1)
model_trainer.load_pre_trained(epoch_count='20')
# %%
################## Train ##################
################## ##### ##################
################## ##### ##################
################## ##### ##################
model_trainer.train(train_input, val_input, epochs=10)
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
