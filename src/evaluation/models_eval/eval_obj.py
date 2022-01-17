

"""
This class uses the test dataset and MC technique for model evaluation.
Steps for MC evaluation a given model is as follows:
() read the eval_config file. This determines what mc experiments need to be run for
which model.
(1) load the testset for a given traffic desity. TestdataObj contains the test data.
() divide the testset into snippets: past states and future states
() scale (with data_obj) and then feed the testset to the model to compute actions
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
# from envs.merge_mc import EnvMergeMC
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
from src.evaluation.models_eval import forward_sim
reload(forward_sim)
from src.evaluation.models_eval.forward_sim import ForwardSim

class MCEVAL():
    eval_config_dir = './src/models/evaluation/config.json'
    def __init__(self):
        self.collections = {} # collection of mc visited states
        self.fs = ForwardSim()

    #     with open('./src/envs/config.json', 'rb') as handle:
    #         config = json.load(handle)
    #     # self.env = EnvMergeMC(config)
    #     self.val_run_name = val_run_name # folder name in which val logs are dumped
    #     self.config = self.read_eval_config()
    #     self.env.metric_collection_mode = True
    #     self.rollout_len = self.config['mc_config']['rollout_len']
    #     self.history_len = self.config['mc_config']['history_len']
    #
    # def read_eval_config(self):
    #     with open(self.eval_config_dir, 'rb') as handle:
    #         config = json.load(handle)
    #     return config

    def update_eval_config(self, model_name):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        mc_config = self.config['mc_config']
        progress_logging = self.config['progress_logging'][model_name]
        progress_logging['last_update'] = dt_string
        progress_logging['episode_in_prog'] = \
                                    f'{self.episode_in_prog}/{mc_config["episodes_n"]}'

        progress_logging['episode'] = self.episode_id

        if self.episode_in_prog == mc_config['episodes_n']:
            self.config['status'] = 'COMPLETE'
        else:
            self.config['status'] = 'IN PROGRESS ...'


        self.config['progress_logging'][model_name] = progress_logging
        with open(self.eval_config_dir, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    # def load_testset(self):


    def assign_neural_vehicle(self, model_name):
        if self.model_vehicle_map[model_name] == 'NeuralVehicle':
            epoch_count = '10'
            from vehicles.neural.neural_vehicle import NeuralVehicle
            self.env.neural_vehicle = NeuralVehicle()
        elif self.model_vehicle_map[model_name] == 'NeuralIDMVehicle':
            epoch_count = '10'
            from vehicles.neural.neural_idm_vehicle import NeuralIDMVehicle
            self.env.neural_vehicle = NeuralIDMVehicle()
        elif self.model_vehicle_map[model_name] == 'LatentMLPVehicle':
            epoch_count = '20'
            from vehicles.neural.latent_mlp_vehicle import LatentMLPVehicle
            self.env.neural_vehicle = LatentMLPVehicle()
        elif self.model_vehicle_map[model_name] == 'MLPVehicle':
            epoch_count = '20'
            from vehicles.neural.mlp_vehicle import MLPVehicle
            self.env.neural_vehicle = MLPVehicle()
        elif self.model_vehicle_map[model_name] == 'LSTMVehicle':
            epoch_count = '20'
            from vehicles.neural.lstm_vehicle import LSTMVehicle
            self.env.neural_vehicle = LSTMVehicle()

        self.env.neural_vehicle.initialize_agent(
                        model_name,
                        epoch_count,
                        self.config['mc_config']['data_id'])

    def dump_mc_logs(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/'+self.val_run_name
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        with open(exp_dir+'/real_collection.pickle', 'wb') as handle:
            pickle.dump(self.real_collection, handle)

        with open(exp_dir+'/rwse_collection.pickle', 'wb') as handle:
            pickle.dump(self.rwse_collection, handle)

        with open(exp_dir+'/runtime.pickle', 'wb') as handle:
            pickle.dump(self.runtime, handle)

        if self.collision_log:
            with open(exp_dir+'/collision_log.pickle', 'wb') as handle:
                pickle.dump(self.collision_log, handle)

    def run_trace(self, trace):
        self.env.initialize_env(self.episode_id)
        tf.random.set_seed(trace) # each trace has a unique tf seed

        time_start = time.time()
        for i in range(0, self.env.trans_time+self.rollout_len):
            self.env.step()
            if self.env.collision_detected:
                collision_id = f'{self.episode_id}_{trace}_'+self.env.collision_vehs
                if collision_id not in self.collision_log:
                    self.collision_log.append(collision_id)

        time_end = time.time()
        runtime = (time_end - time_start)/len(self.env.real_vehicles)

        for veh_id, data_log in self.env.ima_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [self.episode_id, veh_id, trace]
            if not self.episode_id in self.rwse_collection:
                self.rwse_collection[self.episode_id] = {}
            if not veh_id in self.rwse_collection[self.episode_id]:
                self.rwse_collection[self.episode_id][veh_id] = [data_log]
            else:
                # append additional traces
                self.rwse_collection[self.episode_id][veh_id].append(data_log)
        self.runtime.append([self.episode_id, trace, runtime])

    def run_episode(self):
        self.episode_id += 1
        self.episode_in_prog += 1
        np.random.seed(self.episode_id)
        self.env.trans_time = np.random.randint(\
                            self.history_len, self.history_len*2) # controller ==> 'neural'

        for trace in range(self.config['mc_config']['trace_n']):
            self.run_trace(trace)
        for veh_id, data_log in self.env.real_mc_log.items():
            for step_log in data_log:
                step_log[1:1] = [self.episode_id, veh_id, trace]
            if not self.episode_id in self.real_collection:
                self.real_collection[self.episode_id] = {}
            if not veh_id in self.real_collection[self.episode_id]:
                self.real_collection[self.episode_id][veh_id] = [data_log]
            else:
                # in case there are multiple traces per episode
                self.real_collection[self.episode_id][veh_id].append(data_log)

    def initiate_eval(self, model_name):
        self.episode_in_prog = 0
        self.create_empty()
        progress_logging = {}
        self.episode_id = 500
        self.target_episode = self.config['mc_config']['episodes_n'] + \
                                                            self.episode_id

        progress_logging['episode'] = self.episode_id
        progress_logging['episode_in_prog'] = 'NA'
        progress_logging['last_update'] = 'NA'
        self.config['progress_logging'][model_name] = progress_logging

    def load_collections(self, model_name):
        exp_dir = './src/models/experiments/'+model_name+'/'+self.val_run_name
        with open(exp_dir+'/real_collection.pickle', 'rb') as handle:
            self.real_collection = pickle.load(handle)

        with open(exp_dir+'/rwse_collection.pickle', 'rb') as handle:
            self.rwse_collection = pickle.load(handle)

        with open(exp_dir+'/runtime.pickle', 'rb') as handle:
            self.runtime = pickle.load(handle)

        try:
            with open(exp_dir+'/collision_log.pickle', 'rb') as handle:
                self.collision_log = pickle.load(handle)
        except:
            self.collision_log = []

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
            self.episode_id = progress_logging['episode']
            progress_logging['episode_in_prog'] = \
                        f'{self.episode_in_prog}/{mc_config["episodes_n"]}'
            self.target_episode =  self.episode_id + epis_n_left
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
        state_arr_sca[:, 2:] = self.data_obj.applyStateScaler(state_arr_sca[:, 2:])
        target_arr_sca[:, 2:] = self.data_obj.applyActionScaler(target_arr_sca[:, 2:])

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


    def load_test_episode_ids(self, traffic_density):
        assert traffic_density == '' or \
                                        traffic_density == 'high_density' or \
                                        traffic_density == 'medium_density' or \
                                        traffic_density == 'low_density'

        if traffic_density == '':
            dir_path = f'./src/datasets/validation_episodes.csv'
        else:
            dir_path = f'./src/datasets/{traffic_density+"_test_episodes"}.csv'

        test_episodes = np.loadtxt(dir_path, delimiter=',')
        return test_episodes

    # def get_rwse(self, true_states, pred_states):
    #     def
    def run(self):
        # self.model_vehicle_map = self.config['model_vehicle_map']
        # model_names = self.model_vehicle_map.keys()
        #
        # for model_name in model_names:
        #     if self.is_eval_complete(model_name):
        #         continue
        #

        # test_densities = # traffic densities to evaluate the model on
        # for density in test_densities
        # episode_ids = self.load_test_episode_ids(density)
        episode_ids = [2815]

        pred_collection = [] # evental shape: [m scenarios, n traces, time_steps_n, states_n]
        true_collection = [] # evental shape: [m scenarios, 1, time_steps_n, states_n]
        # lin = np.linspace(-3, 3, 20)
        # lin.shape = (20, 1)
        # lin = np.repeat(lin, 21, axis=1)
        # lin.shape = (20, 21, 1)
        np.random.seed(2020)
        tf.random.set_seed(2020) # each trace has a unique tf seed

        for episode_id in episode_ids:
            outputs = self.prep_episode(episode_id=episode_id, splits_n=6)
            test_data, state0s, true_state_snips = outputs
            # true_state_snips: [episode_id, time_stamps, ...]
            self.fs.max_pc = true_state_snips[:, :, self.fs.indx_m['pc']+2].max()
            self.fs.min_pc = true_state_snips[:, :, self.fs.indx_m['pc']+2].min()

            for split_i in range(6):
            # get action plans for this data
                # indxs = [[2, 3], [6, 7], [10, 11], [14, 15]]
                # action_plans = [true_state_snips[split_i:split_i+1, 19:, item] for item in indxs]
                # print(action_plans[0].shape)
                # action_plans = [np.repeat(act[:, :, :], 20, axis=0) for act in action_plans]
                # action_plans[0][:, :, 0:1] += lin

                states = test_data[0][[split_i], :, :]
                conds = [item[[split_i], :, :] for item in test_data[1]]
                # print(conds[0].shape)
                action_plans = self.policy.gen_action_seq(\
                        [states, conds], traj_n=self.traces_n, steps_n=self.pred_step_n)

                state0 = np.repeat(state0s[[split_i], :], self.traces_n, axis=0)
                state_trace = self.fs.forward_sim(state0, action_plans)
                pred_collection.append(state_trace)
                true_collection.append(true_state_snips[[split_i], self.obs_n-1:, :])
        return np.array(true_collection), np.array(pred_collection)

            # self.assign_neural_vehicle(model_name)
            # print('Model being evaluated: ', model_name)
            # while self.episode_id < self.target_episode:
            #     prep
            #     for snip
            #         repeat n times
            #         [m, n, time_steps_n, states_n]
            #
            #
            #     self.run_episode()
            #     self.dump_mc_logs(model_name)
            #     self.update_eval_config(model_name)

class TestdataObj():
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
                "step_size": 1,
                "Note": ""
},
"exp_id": "cae_003",
"Note": "Without guided learning"
}
np.set_printoptions(suppress=True)

eval_obj = MCEVAL()
# data_obj = TestdataObj(config['data_config'])

eval_obj.load_test_episode_ids(traffic_density='high_density')
eval_obj.obs_n = config['data_config']['obs_n']
eval_obj.pred_step_n = 20
eval_obj.step_size = 1
eval_obj.traces_n  = 20

eval_obj.states_arr = data_obj.states_arr
eval_obj.targets_arr = data_obj.targets_arr
eval_obj.data_obj = data_obj.data_obj
# %%

from src.planner import action_policy
reload(action_policy)
from src.planner.action_policy import Policy
policy = Policy(config)
policy.load_model(epoch=20)

eval_obj.policy = policy
eval_obj.policy.action_scaler = data_obj.data_obj.action_scaler
eval_obj.policy.model.dec_model.model_use


time = np.arange(0, 21)
act_coarse = pred_collection[0, :, ::3, 2:4]
act_fine = pred_collection[0, :, :, 2:4]
time[::4]
act_coarse.shape
policy.construct_policy([act_coarse], 7)
np.linspace(0, 2.1, 8)
# %%

for i in range(5):
    plt.plot(time[::4], act_coarse[i, :, 0], color='grey')
    plt.scatter(time[::4], act_coarse[i, :, 0])
    plt.plot(time, act_fine[i, :, 0], color='orange')
    # plt.plot(time, act)

# %%

true_collection, pred_collection = eval_obj.run()

true_collection[0, 0, 1, 2]
pred_collection[0, 0, 1, 2]

pred_collection[0, 1, :, 2]

model_name = 'test'
snips_pred = {}
snips_true = {}
snips_pred[model_name] = pred_collection
snips_true[model_name] = true_collection[:, :, :, 2:]

true_collection.shape
pred_collection.shape
pred_collection.shape
# %%


# %%
for scene_i in range(6):
    plt.figure()
    plt.plot(true_collection[scene_i, 0, :, 3], color='red')
    for i in range(20):
        plt.plot(pred_collection[scene_i, i, :, 1], color='grey')

# %%
for scene_i in range(6):
    plt.figure()
    plt.plot(true_collection[scene_i, 0, :, 5], color='red')
    for i in range(20):
        plt.plot(pred_collection[scene_i, i, :, 3], color='grey')
# %%


# %%
def get_trace_err(pred_traces, true_trace):
    """
    Input shpae [trace_n, steps_n]
    Return shape [1, steps_n]
    """
    # mean across traces (axis=0)
    return np.mean((pred_traces - true_trace)**2, axis=0)

def get_veh_err(index, model_name):
    """
    Input shpae [veh_n, trace_n, steps_n, state_index]
    Return shape [veh_n, steps_n]
    """
    posx_true = snips_true[model_name][:,:,:,index]
    posx_pred = snips_pred[model_name][:,:,:,index]

    vehs_err_arr = [] # vehicles error array
    veh_n = snips_true[model_name].shape[0]
    for i in range(veh_n):
        vehs_err_arr.append(get_trace_err(posx_pred[i, :, :], posx_true[i, :, :]))
    return np.array(vehs_err_arr)

def get_rwse(vehs_err_arr):
    # mean across all snippets (axis=0)
    return np.mean(vehs_err_arr, axis=0)**0.5
# %%


time_vals = np.linspace(0, 2.1, 21)

fig = plt.figure(figsize=(6, 4))
position_axis = fig.add_subplot(211)
speed_axis = fig.add_subplot(212)
fig.subplots_adjust(hspace=0.1)
# for model_name in model_names:

for model_name in ['test']:
    vehs_err_arr = get_veh_err(0, model_name)
    vehs_err_arr.shape


    error_total = get_rwse(vehs_err_arr)
    speed_axis.plot(time_vals, error_total)
error_total.shape
speed_axis.set_ylabel('RWSE speed ($ms^{-1}$)')
speed_axis.set_xlabel('Time horizon (s)')
speed_axis.minorticks_off()
# speed_axis.set_ylim(0, 2)
speed_axis.set_yticks([0, 1, 2, 3])
# speed_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=5)

# %%

# %%

test_data, state0s, true_state_snips = eval_obj.prep_episode(episode_id=2815, splits_n=6)
states, conds = test_data
conds[0].shape
state0s.shape
state0s[0, :]
states[0, 0, :]
states[0, -1, :]
states[0, :, :].shape
states.shape
# %%
from importlib import reload
import matplotlib.pyplot as plt


true_state_snips.shape
forw.indx_m
forw.indx_y
forw.indx_f
forw.indx_fadj
indxs = np.array([[2, 3], [6, 7], [10, 11], [14, 15]])+2
indxs = indxs.tolist()
action_plans = [true_state_snips[:, 19:, item] for item in indxs]
action_plans = [a + 1 for a in action_plans]
action_plans[0].shape
action_plans[0][0, :, :]
state0s.shape

true_state_snips[:, :, 3].max()
state_trace = forw.forward_sim(state0s[:, 2:], action_plans)
state_trace.shape
true_state_snips[:, 19, 1]
true_state_snips.shape
true_state_snips[0:1, :, [2, 3]].shape
true_state_snips[[0], :, :].shape
true_state_snips[[], :, [2, 3]].shape
# %%

action_plans[0].shape
# plt.plot(action_plans[1][0, :, 0])
# plt.plot(state_trace[0, :, 0])
state0s[0, :]
# plt.plot(state_trace[0, :, 3])
true_state_snips[0, 19, :]
plt.plot(true_state_snips[0, 19:, 2], color='red')
plt.plot(state_trace[0, :, 0])
true_state_snips[0, 19:, 2].shape
state_trace[0, :, 2].shape
# %%

states.shape
states[1, :, :]
(states - states[0:1, :, :])[0]
(states - states[0:1, :, :])[1, :, :]


np.array(true_state_snips).shape
states[0, :, 1]
np.array(true_state_snips)[0, :, 1]
states[2, :, 1]
conds[0][2, :, 1]
 
