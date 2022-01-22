import os
# from planner import forward_sim
# reload(forward_sim)
from planner.forward_sim import ForwardSim
from planner.state_indexs import StateIndxs
from evaluation.eval_data_obj import EvalDataObj
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
    def __init__(self, val_run_name=None):
        self.collections = {} # collection of mc visited states
        self.fs = ForwardSim()
        self.indxs = StateIndxs()
        self.data_obj = EvalDataObj()
        self.val_run_name = val_run_name
        self.loadScalers() # will set the scaler attributes

    def loadScalers(self):
        with open('./src/datasets/'+'state_scaler', 'rb') as f:
            self.state_scaler = pickle.load(f)
        with open('./src/datasets/'+'action_scaler', 'rb') as f:
            self.action_scaler = pickle.load(f)

    def read_eval_config(self, config_name):
        self.eval_config_dir = './src/evaluation/models_eval/'+ config_name +'.json'
        with open(self.eval_config_dir, 'rb') as handle:
            self.config = json.load(handle)
        self.traces_n = self.config['mc_config']['traces_n']
        self.splits_n = self.config['mc_config']['splits_n']

    def read_model_config(self, model_name):
        exp_dir = './src/models/experiments/'+model_name
        with open(exp_dir+'/'+'config.json', 'rb') as handle:
            config = json.load(handle)

        self.obs_n = config['data_config']['obs_n']
        self.step_size = config['data_config']['step_size']
        self.pred_step_n = np.ceil(20/self.step_size).astype('int')
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
        exp_dir = './src/models/experiments/'+model_name+'/'+self.val_run_name
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
        exp_dir = './src/models/experiments/'+model_name+'/'+self.val_run_name
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

        if traj_len > 30:
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

    def prep_episode(self, episode_id):
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
        states, targs, conds = self.obsSequence(state_arr_sca, target_arr_sca)
        if states.shape[0] < self.splits_n:
            return
        random_snippets = np.random.choice(range(states.shape[0]), self.splits_n, replace=False)

        states = states[random_snippets, :, 2:]
        targs = [targ[random_snippets, :, 2:] for targ in targs]
        conds = [cond[random_snippets, :, 2:] for cond in conds]

        true_state_snips = []
        for start_step in random_snippets:
            true_state_snips.append(state_arr[start_step:start_step + 40, :])

        test_data = [states[:, :, :], conds] # to feed to model
        return test_data, np.array(true_state_snips)[:, :, :]

    def run_episode(self, episode_id):
        # test_densities = # traffic densities to evaluate the model on
        # for density in test_densities
        pred_collection = [] # evental shape: [m scenarios, n traces, time_steps_n, states_n]
        true_collection = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]
        np.random.seed(2020)
        tf.random.set_seed(2020) # each trace has a unique tf seed
        outputs = self.prep_episode(episode_id=episode_id)
        if not outputs:
            return true_collection, pred_collection
        test_data, true_state_snips = outputs
        # true_state_snips: [episode_id, time_stamps, ...]
        self.fs.max_pc = true_state_snips[:, :, self.indxs.indx_m['pc']+2].max()
        self.fs.min_pc = true_state_snips[:, :, self.indxs.indx_m['pc']+2].min()

        for split_i in range(self.splits_n):
        # get action plans for this scene
            states = test_data[0][[split_i], :, :]
            conds = [item[[split_i], :, :] for item in test_data[1]]
            true_trace = true_state_snips[[split_i], :, :]
            pred_trace = self.get_predicted_trace(states, conds, true_trace)
            pred_collection.append(pred_trace)
            true_collection.append(true_trace)

        return true_collection, pred_collection

    def get_predicted_trace(self, states, conds, true_trace):
        true_trace_history = np.repeat(\
                true_trace[:, :self.obs_n-1, 2:], self.traces_n, axis=0)

        gen_actions = self.policy.gen_action_seq(\
                            [states, conds], traj_n=self.traces_n)

        bc_ders = self.policy.get_boundary_condition(true_trace_history)
        action_plans = self.policy.construct_policy(gen_actions, bc_ders, self.traces_n)
        state_0 = true_trace_history[:, -1, :]
        pred_trace = self.fs.forward_sim(state_0, action_plans)
        return pred_trace

    def load_policy(self, model_name):
        model_config = self.read_model_config(model_name)
        model_type, epoch = self.config['model_map'][model_name]
        if model_type == 'CAE':
            from planner.action_policy import Policy
            self.policy = Policy()

        if model_type == 'MLP':
            exp_dir = './src/models/experiments/'+model_name
            exp_path = f'{exp_dir}/model_epo{epoch}'

            from models.core.mlp import MLP
            self.policy = MLP(model_config)
            self.policy.load_weights(exp_path).expect_partial()

        if model_type == 'LSTM':
            exp_dir = './src/models/experiments/'+model_name
            exp_path = f'{exp_dir}/model_epo{epoch}'

            from models.core.lstm import LSTMEncoder
            self.policy = LSTMEncoder(model_config)
            self.policy.load_weights(exp_path).expect_partial()

    def run(self):
        model_names = self.config['model_map'].keys()
        self.states_arr, self.targets_arr = self.data_obj.load_val_data()
        for model_name in model_names:
            print('Model being evaluated: ', model_name)
            if self.is_eval_complete(model_name):
                print('Oops - this model is already evaluated.')
                continue
            else:
                self.load_policy(model_name)
                episode_ids = self.data_obj.load_test_episode_ids('')
                i = self.episode_in_prog
                while self.episode_in_prog < self.target_episode_count:
                    true_collection, pred_collection = self.run_episode(\
                                                    episode_ids[i])

                    if true_collection:
                        self.true_collections.extend(true_collection)
                        self.pred_collections.extend(pred_collection)
                        self.episode_in_prog += 1

                        self.dump_mc_logs(model_name)
                        self.update_eval_config(model_name)
                    i += 1
