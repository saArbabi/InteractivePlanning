"""
- Outputs training set and validation set for a particular experiment prior to
start of training.
"""
from collections import deque
import numpy as np
from sklearn.utils import shuffle
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
import dill
# %%
def read_data():
    global all_state_arr, all_target_arr
    global training_episodes, validation_episodes, test_episodes

    all_state_arr = np.loadtxt('./src/datasets/states_arr.csv', delimiter=',')
    all_target_arr = np.loadtxt('./src/datasets/targets_arr.csv', delimiter=',')

    training_episodes = np.loadtxt('./src/datasets/training_episodes.csv', delimiter=',')
    validation_episodes = np.loadtxt('./src/datasets/validation_episodes.csv', delimiter=',')

read_data()
# %%
class DataPrep():
    random.seed(2020)

    def __init__(self, config, dirName):
        self.config = config['data_config']
        self.obs_n = self.config["obs_n"]
        self.pred_step_n = self.config["pred_step_n"]
        self.step_size = self.config["step_size"]
        self.dirName = dirName
        os.mkdir(dirName)
        self.loadScalers() # will set the scaler attributes

    def obsSequence(self, state_arr, target_arr):
        actions = [target_arr[:, n:n+2] for n in range(8)[::2] ]
        traj_len = len(state_arr)

        if traj_len > 20:
            prev_states = deque(maxlen=self.obs_n)
            for i in range(traj_len):
                prev_states.append(state_arr[i])

                if len(prev_states) == self.obs_n:
                    indx = np.arange(i, i+(self.pred_step_n+1)*self.step_size, self.step_size)
                    indx = indx[indx<traj_len]
                    if indx.size < 2:
                        break

                    seq_len = len(indx)-1
                    if seq_len not in self.targs:
                        self.targs[seq_len] = [[] for n in range(4)]
                        self.conds[seq_len] = [[] for n in range(4)]
                        self.states[seq_len] = []

                    self.states[seq_len].append(np.array(prev_states))
                    for n in range(4):
                        # jerk = actions[n][indx[1:]]-actions[n][indx[:-1]]
                        self.targs[seq_len][n].append(actions[n][indx[1:]])
                        self.conds[seq_len][n].append(actions[n][indx[:-1]])

    def get_episode_arr(self, episode_id):
        state_arr = all_state_arr[all_state_arr[:, 0] == episode_id]
        target_arr = all_target_arr[all_target_arr[:, 0] == episode_id]
        return state_arr[:, 1:], target_arr[:, 1:]

    def applyStateScaler(self, _arr):
        _arr[:, :-4] = self.state_scaler.transform(_arr[:, :-4])
        return _arr

    def applyActionScaler(self, _arr):
        return self.action_scaler.transform(_arr)

    def loadScalers(self):
        with open('./src/datasets/'+'state_scaler', 'rb') as f:
            self.state_scaler = pickle.load(f)
        with open('./src/datasets/'+'action_scaler', 'rb') as f:
            self.action_scaler = pickle.load(f)

    def episode_prep(self, episode_id):
        """
        :Return: x, y arrays for model training.
        """
        state_arr, target_arr = self.get_episode_arr(episode_id)
        state_arr = self.applyStateScaler(state_arr)
        target_arr = self.applyActionScaler(target_arr)
        self.obsSequence(state_arr, target_arr)

    def shuffle(self, data_dict, type):
        for seq_n in range(1, self.pred_step_n+1):
            if type=='targs':
                # targets and conditionals
                data_dict[seq_n] = [np.array(shuffle(data_dict[seq_n][n], \
                                            random_state=2020)) for n in range(4)]
            elif type=='states':
                # states
                data_dict[seq_n] = np.array(shuffle(data_dict[seq_n], random_state=2020))
        return data_dict

    def pickler(self, episode_type):
        self.states = self.shuffle(self.states, 'states')
        self.targs = self.shuffle(self.targs, 'targs')
        self.conds = self.shuffle(self.conds, 'targs')

        if episode_type == 'validation_episodes':
            with open(self.dirName+'/states_val', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_val', "wb") as f:
                pickle.dump(self.targs, f)

            with open(self.dirName+'/conditions_val', "wb") as f:
                pickle.dump(self.conds, f)

            delattr(self, 'states')
            delattr(self, 'targs')
            delattr(self, 'conds')

        elif episode_type == 'training_episodes':
            with open(self.dirName+'/states_train', "wb") as f:
                pickle.dump(self.states, f)

            with open(self.dirName+'/targets_train', "wb") as f:
                pickle.dump(self.targs, f)

            with open(self.dirName+'/conditions_train', "wb") as f:
                pickle.dump(self.conds, f)

            delattr(self, 'states')
            delattr(self, 'targs')
            delattr(self, 'conds')

    def data_prep(self, episode_type=None):
        if not episode_type:
            raise ValueError("Choose training_episodes or validation_episodes")
        self.states = {}
        self.targs = {}
        self.conds = {}

        if episode_type == 'training_episodes':
            for episode_id in training_episodes:
                # print(f'prepping {episode_type} - episode: {episode_id}')
                self.episode_prep(episode_id)

        elif episode_type == 'validation_episodes':
            for episode_id in validation_episodes:
                # print(f'prepping {episode_type} - episode: {episode_id}')
                self.episode_prep(episode_id)

        self.pickler(episode_type)
