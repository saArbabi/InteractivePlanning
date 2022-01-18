import numpy as np
import pickle
import os
import json
import dill

class EvalDataObj():
    dir_path = './src/datasets/preprocessed/'
    def __init__(self, data_config):
        self.load_test_data(data_config) # load test_data and validation data

    def load_test_data(self, data_config):
        config_names = os.listdir(self.dir_path+'config_files')
        for config_name in config_names:
            with open(self.dir_path+'config_files/'+config_name, 'r') as f:
                config = json.load(f)
            if config == data_config:
                with open(self.dir_path+config_name[:-5]+'/'+'data_obj', 'rb') as f:
                    self.data_obj = dill.load(f, ignore=True)

        self.states_arr = np.loadtxt('./src/datasets/states_arr.csv', delimiter=',')
        self.targets_arr = np.loadtxt('./src/datasets/targets_arr.csv', delimiter=',')

    def load_test_episode_ids(self, traffic_density):
        assert traffic_density == '' or \
                                        traffic_density == 'high_density' or \
                                        traffic_density == 'medium_density' or \
                                        traffic_density == 'low_density'

        if traffic_density == '':
            dir_path = f'./src/datasets/validation_episodes.csv'
        else:
            dir_path = f'./src/datasets/{traffic_density+"_test_episodes"}.csv'

        self.test_episodes = np.loadtxt(dir_path, delimiter=',')
