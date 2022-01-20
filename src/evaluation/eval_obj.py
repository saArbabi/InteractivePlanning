

"""
This class uses the test dataset and MC technique for model evaluation.
Steps for MC evaluation a given model is as follows:
() read the eval_config file. This determines what mc experiments need to be run for
which model.
(1) load the testset for a given traffic desity. EvalDataObj contains the test data.
() divide the testset into snippets: past states and future states
() scale and then feed the testset to the model to compute actions
() turn the plans to states of interest (forward_sim)
() dump collected states for all the vehicels of interest in their respective model folders.
    depending on the model, different rwse metrics are collected
() you can now compute rwse for different models in ./publication/quantitative


    - rwse
        - high denstiy
            - veh_m
                - long. speed
                - lat. speed
            - veh_y
            - veh_f
            - veh_fadj
        - medium denstiy
        - random denstiy (for evaluating influence of sequence length and step size )

Note: random seeds are set to ensure repeatable MC runs.

"""

import tensorflow as tf
import time
from datetime import datetime
import os
import numpy as np
import pickle
import dill
from collections import deque
from importlib import reload
import matplotlib.pyplot as plt
import json
from src.planner import forward_sim
reload(forward_sim)
from src.planner.forward_sim import ForwardSim
from src.planner.state_indexs import StateIndxs
from src.evaluation.eval_data_obj import EvalDataObj

class MCEVAL():
    eval_config_dir = './src/evaluation/models_eval/config.json'
    def __init__(self):
        self.collections = {} # collection of mc visited states
        self.fs = ForwardSim()
        self.indxs = StateIndxs()
        self.data_obj = EvalDataObj()
        self.config = self.read_eval_config()
        self.traces_n = self.config['mc_config']['traces_n']
        self.loadScalers() # will set the scaler attributes

    def loadScalers(self):
        with open('./src/datasets/'+'state_scaler', 'rb') as f:
            self.state_scaler = pickle.load(f)
        with open('./src/datasets/'+'action_scaler', 'rb') as f:
            self.action_scaler = pickle.load(f)

    def read_eval_config(self):
        with open(self.eval_config_dir, 'rb') as handle:
            config = json.load(handle)
        return config

    def update_eval_config(self, model_name):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        mc_config = self.config['mc_config']
        progress_logging = self.config['progress_logging'][model_name]
        progress_logging['last_update'] = dt_string
        progress_logging['episode_in_prog'] = \
                                    f'{self.episode_in_prog}/{mc_config["episodes_n"]}'


        if self.episode_in_prog == mc_config['episodes_n']:
            self.config['status'] = 'COMPLETE'
        else:
            self.config['status'] = 'IN PROGRESS ...'

        self.config['progress_logging'][model_name] = progress_logging
        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def dump_mc_logs(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/eval'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        with open(exp_dir+'/true_collections.pickle', 'wb') as handle:
            pickle.dump(self.true_collections, handle)

        with open(exp_dir+'/pred_collections.pickle', 'wb') as handle:
            pickle.dump(self.pred_collections, handle)

    def initiate_eval(self, model_name):
        self.true_collections = []
        self.pred_collections = []
        progress_logging = {}
        self.target_episode_count = self.config['mc_config']['episodes_n']
        self.episode_in_prog = 0
        progress_logging['episode_in_prog'] = 'NA'
        progress_logging['last_update'] = 'NA'
        self.config['progress_logging'][model_name] = progress_logging

    def load_collections(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/eval'
        with open(exp_dir+'/true_collections.pickle', 'rb') as handle:
            self.true_collections = pickle.load(handle)

        with open(exp_dir+'/pred_collections.pickle', 'rb') as handle:
            self.pred_collections = pickle.load(handle)

    def is_eval_complete(self, model_name):
        """Check if this model has been fully evaluated.
        """
        if not model_name in self.config['progress_logging']:
            self.initiate_eval(model_name)
            return False

        progress_logging = self.config['progress_logging'][model_name]
        mc_config = self.config['mc_config']
        epis_n_left = 0 # remaining episodes ot compelte

        episode_in_prog = progress_logging['episode_in_prog']
        episode_in_prog = episode_in_prog.split('/')
        self.episode_in_prog = int(episode_in_prog[0])
        epis_n_left = mc_config['episodes_n'] - self.episode_in_prog
        if epis_n_left == 0:
            return True
        else:
            self.load_collections(model_name)
            progress_logging['episode_in_prog'] = \
                        f'{self.episode_in_prog}/{mc_config["episodes_n"]}'
            self.target_episode_count =  mc_config['episodes_n']
            self.update_eval_config(model_name)
            return False

    def get_episode_arr(self, episode_id):
        state_arr = self.states_arr[self.states_arr[:, 0] == episode_id]
        target_arr = self.targets_arr[self.targets_arr[:, 0] == episode_id]
        return state_arr, target_arr

    def obsSequence(self, state_arr, target_arr):
        """
        Rolling sequencing
        """
        actions = [target_arr[:, np.r_[0:2, n:n+2]] for n in range(2, 10)[::2]]
        traj_len = len(state_arr)
        states = []
        targs = [[] for n in range(4)]
        conds = [[] for n in range(4)]

        if traj_len > 20:
            prev_states = deque(maxlen=self.obs_n)
            for i in range(traj_len):
                prev_states.append(state_arr[i])

                if len(prev_states) == self.obs_n:
                    indx = np.arange(i, i+(self.pred_step_n+1)*self.step_size, self.step_size)
                    indx = indx[indx<traj_len]
                    if indx.size != self.pred_step_n+1:
                        break

                    states.append(np.array(prev_states))
                    for n in range(4):
                        targs[n].append(actions[n][indx[1:]])
                        conds[n].append(actions[n][indx[:-1]])

        targs = [np.array(targ) for targ in targs]
        conds = [np.array(cond) for cond in conds]
        return np.array(states), targs, conds

    def prep_episode(self, episode_id, splits_n):
        """
        Split the episode sequence into splits_n.
        Returns test_data to feed the model and true_state_snips for model evaluation
        """
        state_arr, target_arr = self.get_episode_arr(episode_id)
        episode_len = state_arr.shape[0]
        time_stamps = range(episode_len)
        state_arr = np.insert(state_arr, 1, time_stamps, axis=1)
        target_arr = np.insert(target_arr, 1, time_stamps, axis=1)
        state_arr_sca = state_arr.copy()
        target_arr_sca = target_arr.copy()
        state_arr_sca[:, 2:-4] = self.state_scaler.transform(state_arr_sca[:, 2:-4])
        target_arr_sca[:, 2:] = self.action_scaler.transform(target_arr_sca[:, 2:])

        splits_n = 6
        states, targs, conds = self.obsSequence(state_arr_sca, target_arr_sca)
        random_snippets = np.random.choice(range(states.shape[0]), splits_n, replace=False)

        start_steps = states[random_snippets, 0, 1].astype('int')
        end_steps = targs[0][random_snippets, -1, 1].astype('int')

        states = states[random_snippets, :, 2:]
        targs = [targ[random_snippets, :, 2:] for targ in targs]
        conds = [cond[random_snippets, :, 2:] for cond in conds]

        true_state_snips = []
        state0s = []

        for start_step, end_step in zip(start_steps, end_steps):
            true_state_snips.append(state_arr[start_step:end_step+1, :])
            state0s.append(state_arr[start_step+self.obs_n-1, 2:])

        test_data = [states[:, :, :], conds] # to feed to model
        return test_data, np.array(state0s)[:, :], np.array(true_state_snips)[:, :, :]

    def run_episode(self, episode_id):
        # test_densities = # traffic densities to evaluate the model on
        # for density in test_densities
        # episode_ids = self.data_obj.load_test_episode_ids(density)

        pred_collection = [] # evental shape: [m scenarios, n traces, time_steps_n, states_n]
        true_collection = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]
        np.random.seed(2020)
        tf.random.set_seed(2020) # each trace has a unique tf seed
        outputs = self.prep_episode(episode_id=episode_id, splits_n=6)
        test_data, state0s, true_state_snips = outputs
        # true_state_snips: [episode_id, time_stamps, ...]
        self.fs.max_pc = true_state_snips[:, :, self.indxs.indx_m['pc']+2].max()
        self.fs.min_pc = true_state_snips[:, :, self.indxs.indx_m['pc']+2].min()

        for split_i in range(6):
        # get action plans for this scene
            states = test_data[0][[split_i], :, :]
            conds = [item[[split_i], :, :] for item in test_data[1]]
            gen_actions = self.policy.gen_action_seq(\
                                [states, conds], traj_n=self.traces_n)
            action_plans = self.policy.construct_policy(gen_actions, states)

            state0 = np.repeat(state0s[[split_i], :], self.traces_n, axis=0)
            state_trace = self.fs.forward_sim(state0, action_plans)
            pred_collection.append(state_trace)
            true_collection.append(true_state_snips[[split_i], :, :])

        return true_collection, pred_collection

    def run(self):
        self.model_map = self.config['model_map']
        model_names = self.model_map.keys()
        self.states_arr, self.targets_arr = self.data_obj.load_val_data()

        for model_name in model_names:
            if self.is_eval_complete(model_name):
                continue
            self.policy.load_model(model_name)
            print('Model being evaluated: ', model_name)
            episode_ids = [129]

            while self.episode_in_prog < self.target_episode_count:
                true_collection, pred_collection = self.run_episode(\
                                                episode_ids[self.episode_in_prog])

                self.true_collections.extend(true_collection)
                self.pred_collections.extend(pred_collection)
                self.episode_in_prog += 1

                self.dump_mc_logs(model_name)
                self.update_eval_config(model_name)
