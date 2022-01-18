import numpy as np
import pickle
import os
import json
import dill

class EvalDataObj():
    dir_path = './src/datasets/preprocessed/'

    def load_data_obj(self, data_config):
            config_names = os.listdir(self.dir_path+'config_files')
            for config_name in config_names:
                with open(self.dir_path+'config_files/'+config_name, 'r') as f:
                    data_config_i = json.load(f)
                if data_config_i == data_config:
                    with open(self.dir_path+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                        data_obj = dill.load(f, ignore=True)
            return data_obj

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
