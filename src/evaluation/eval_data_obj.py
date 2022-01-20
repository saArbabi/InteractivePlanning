import numpy as np
import pickle
import os
import json
import dill

class EvalDataObj():
    dir_path = './src/datasets/preprocessed/'

    def load_val_data(self):
        states_val_arr = np.loadtxt('./src/datasets/states_val_arr.csv', delimiter=',')
        targets_val_arr = np.loadtxt('./src/datasets/targets_val_arr.csv', delimiter=',')
        return states_val_arr, targets_val_arr

    def load_test_episode_ids(self, traffic_density):
        assert traffic_density == '' or \
                                        traffic_density == 'high_density' or \
                                        traffic_density == 'medium_density' or \
                                        traffic_density == 'low_density'

        if traffic_density == '':
            dir_path = f'./src/datasets/validation_episodes.csv'
        else:
            dir_path = f'./src/datasets/{traffic_density+"_val_episodes"}.csv'
        return np.loadtxt(dir_path, delimiter=',')
