"""
This module uses the test dataset and MC simulations for model evaluation.
Steps for MC evaluation a given model is as follows:
    (1) read the eval_config file. This determines what mc experiments need to be run for
    which model.
    (2) load the testset for a given traffic desity. EvalDataObj contains the test data.
    (3) divide the testset into snippets: past states and future states
    (4) scale and then feed the testset to the model to compute actions
    (5) turn the plans to states of interest (forward_sim)
    (6) dump collected states for all the vehicels of interest in their respective model folders.
        depending on the model, different rwse metrics are collected
    (7) you can now compute rwse for different models in ./publication/quantitative

"""
import os
# from planner import forward_sim
# reload(forward_sim)
from planner.forward_sim import ForwardSim
from planner.state_indexs import StateIndxs
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
import pickle
import dill
from collections import deque
from importlib import reload
import matplotlib.pyplot as plt
import json


class MCEVALMultiStep():
    def __init__(self, config=None):
        self.collections = {} # collection of mc visited states
        self.fs = ForwardSim()
        self.indxs = StateIndxs()

        if config:
            self.eval_config = config
            self.traces_n = self.eval_config['mc_config']['traces_n']
            self.splits_n = self.eval_config['mc_config']['splits_n']

    def read_model_config(self, model_name):
        exp_dir = './src/models/experiments/'+model_name
        with open(exp_dir+'/'+'config.json', 'rb') as handle:
            config = json.load(handle)

        self.obs_n = config['data_config']['obs_n']
        self.step_size = config['data_config']['step_size']
        self.pred_step_n = np.ceil(20/self.step_size).astype('int')
        return config

    def update_eval_config(self):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        mc_config = self.eval_config['mc_config']
        progress_logging = self.eval_config['progress_logging'][self.model_run_name]
        progress_logging['last_update'] = dt_string
        progress_logging['current_episode_count'] = \
                                    f'{self.current_episode_count}/{mc_config["episodes_n"]}'

        progress_logging['episode_in_prog'] = self.episode_in_prog
        if self.current_episode_count == mc_config['episodes_n']:
            self.eval_config['status'] = 'COMPLETE'
        else:
            self.eval_config['status'] = 'IN PROGRESS ...'

        self.eval_config['progress_logging'][self.model_run_name] = progress_logging
        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.eval_config, f, ensure_ascii=False, indent=4)

    def dump_mc_logs(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/'+self.mc_run_name
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        with open(exp_dir+'/true_collections.pickle', 'wb') as handle:
            pickle.dump(self.true_collections, handle)

        with open(exp_dir+'/pred_collections.pickle', 'wb') as handle:
            pickle.dump(self.pred_collections, handle)

    def initiate_eval(self):
        self.true_collections = []
        self.pred_collections = []
        progress_logging = {}
        self.target_episode_count = self.eval_config['mc_config']['episodes_n']
        self.current_episode_count = 0
        self.episode_in_prog = self.episode_ids[0]
        progress_logging['episode_in_prog'] = self.episode_in_prog
        progress_logging['current_episode_count'] = 'NA'
        progress_logging['last_update'] = 'NA'
        self.eval_config['progress_logging'][self.model_run_name] = progress_logging

    def load_collections(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/'+self.mc_run_name
        with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
            self.true_collections = pickle.load(handle)

        with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
            self.pred_collections = pickle.load(handle)

    def is_eval_complete(self, model_name):
        """Check if this model has been fully evaluated.
        """
        if not self.model_run_name in self.eval_config['progress_logging']:
            self.initiate_eval()
            return False

        progress_logging = self.eval_config['progress_logging'][self.model_run_name]
        mc_config = self.eval_config['mc_config']
        epis_n_left = 0 # remaining episodes ot compelte
        self.current_episode_count = int(progress_logging['current_episode_count'].split('/')[0])
        self.episode_in_prog = progress_logging['episode_in_prog']
        epis_n_left = mc_config['episodes_n'] - self.current_episode_count
        if epis_n_left == 0:
            return True
        else:
            self.load_collections(model_name)
            self.target_episode_count = mc_config['episodes_n']
            self.update_eval_config()
            return False

    def get_episode_arr(self, episode_id):
        state_arr = self.states_arr[self.states_arr[:, 0] == episode_id]
        target_arr = self.targets_arr[self.targets_arr[:, 0] == episode_id]
        time_stamps = range(state_arr.shape[0])
        state_arr = np.insert(state_arr, 1, time_stamps, axis=1)
        target_arr = np.insert(target_arr, 1, time_stamps, axis=1)
        return state_arr, target_arr

    def obsSequence(self, state_arr, target_arr):
        """
        Rolling sequencing. Adapted from the DataPrep class so that all test
        scenarios have aligned initial state
        (to ensure fair comparison between different models).
        """
        actions = [target_arr[:, np.r_[0:2, n:n+2]] for n in range(2, 10)[::2]]
        traj_len = len(state_arr)
        states = []
        conds = [[] for n in range(4)]

        prev_states = deque(maxlen=self.obs_n)
        for i in range(traj_len - 30):
            prev_states.append(state_arr[i])

            if len(prev_states) == self.obs_n:
                indx = np.arange(i, i + 30, 1)
                indx = indx[::self.step_size][:self.pred_step_n+1]
                states.append(np.array(prev_states))
                # print(indx)
                for n in range(4):
                    conds[n].append(actions[n][indx[:-1]])

        conds = [np.array(cond) for cond in conds]
        return np.array(states), conds

    def sequence_episode(self, state_arr, target_arr):
        state_arr_sca = state_arr.copy()
        target_arr_sca = target_arr.copy()
        state_arr_sca[:, 2:-4] = self.state_scaler.transform(state_arr_sca[:, 2:-4])
        target_arr_sca[:, 2:] = self.action_scaler.transform(target_arr_sca[:, 2:])
        states, conds = self.obsSequence(state_arr_sca, target_arr_sca)
        return states, conds

    def rand_split_episode(self, state_arr, states, conds):
        """
        Split the episode sequence into splits_n.
        Returns test_data to feed the model and true_state_snips for model evaluation
        """
        random_snippets = np.random.choice(range(states.shape[0]), self.splits_n, replace=False)
        # print(random_snippets)
        states = states[random_snippets, :, 2:]
        conds = [cond[random_snippets, :, 2:] for cond in conds]

        true_state_snips = []
        for start_step in random_snippets:
            true_state_snips.append(state_arr[start_step:start_step + 40, :])

        test_data = [states, conds] # to feed to model
        return test_data, np.array(true_state_snips)


    def run_episode(self, episode_id):
        # test_densities = # traffic densities to evaluate the model on
        np.random.seed(episode_id)
        tf.random.set_seed(episode_id) # each trace has a unique tf seed
        state_arr, target_arr = self.get_episode_arr(episode_id)
        states, conds = self.sequence_episode(state_arr, target_arr)
        if states.shape[0] < self.splits_n:
            return
        test_data, true_state_snips = self.rand_split_episode(state_arr, states, conds)
        # true_state_snips: [episode_id, time_stamps, ...]
        self.fs.max_pc = true_state_snips[:, :, self.indxs.indx_m['pc']+2].max()
        self.fs.min_pc = true_state_snips[:, :, self.indxs.indx_m['pc']+2].min()

        pred_collection = [] # evental shape: [m scenarios, n traces, time_steps_n, states_n]
        true_collection = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]
        for split_i in range(self.splits_n):
            states = test_data[0][[split_i], :, :]
            conds = [item[[split_i], :, :] for item in test_data[1]]
            true_trace = true_state_snips[[split_i], :, :]
            pred_trace = self.get_predicted_trace(states, conds, true_trace)
            pred_collection.append(pred_trace)
            true_collection.append(true_trace)

        return true_collection, pred_collection

    def get_predicted_trace(self, states, conds, true_trace):
        trace_history = np.repeat(\
                true_trace[:, :self.obs_n, 2:], self.traces_n, axis=0)

        states = np.repeat(states, self.traces_n, axis=0)
        conds = [np.repeat(cond, self.traces_n, axis=0) for cond in conds]

        _gen_actions, _ = self.policy.cae_inference([states, conds])
        gen_actions = self.policy.gen_action_seq(_gen_actions, conds)
        bc_ders = self.policy.get_boundary_condition(trace_history)
        action_plans = self.policy.get_pred_vehicle_plans(gen_actions, bc_ders)
        state_0 = trace_history[:, -1, :]
        pred_trace = self.fs.forward_sim(state_0, action_plans)
        return pred_trace

    def load_policy(self, model_name, model_type, epoch):
        model_config = self.read_model_config(model_name)
        if model_type == 'CAE':
            from planner.action_policy import Policy
            policy = Policy()
            policy.load_model(model_config, epoch)
            try:
                policy.traj_n = self.traces_n
            except:
                pass


        if model_type == 'MLP':
            exp_dir = './src/models/experiments/'+model_name
            exp_path = f'{exp_dir}/model_epo{epoch}'

            from models.core.mlp import MLP
            policy = MLP(model_config)
            policy.load_weights(exp_path).expect_partial()

        if model_type == 'LSTM':
            exp_dir = './src/models/experiments/'+model_name
            exp_path = f'{exp_dir}/model_epo{epoch}'

            from models.core.lstm import LSTMEncoder
            policy = LSTMEncoder(model_config)
            policy.load_weights(exp_path).expect_partial()
        return policy

    def run(self, model_name):
        print('Model being evaluated: ', model_name)
        if self.is_eval_complete(model_name):
            print('Oops - this model is already evaluated.')
            pass
        else:
            model_type, epoch, _ = self.eval_config['model_map'][model_name]
            self.model_type = model_type

            self.policy = self.load_policy(model_name, model_type, epoch)
            i = np.where(self.episode_ids == self.episode_in_prog)[0]
            i += 1 if self.current_episode_count > 0 else i # start with the next episode
            while self.current_episode_count < self.target_episode_count:
                self.episode_in_prog = int(self.episode_ids[i])
                collections = self.run_episode(self.episode_in_prog)
                if collections:
                    true_collection, pred_collection = collections
                    self.true_collections.extend(true_collection)
                    self.pred_collections.extend(pred_collection)
                    self.current_episode_count += 1

                    self.dump_mc_logs(model_name)
                    self.update_eval_config()
                i += 1
