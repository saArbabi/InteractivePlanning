import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
import dill
from planner.state_indexs import StateIndxs
from publication.scene_evolution.vehicles import HumanVehicle, CAEVehicle

class Env():
    def __init__(self, state_arr):
        self.indxs = StateIndxs()
        self.lane_width = state_arr[:, self.indxs.indx_m['pc']].max()*2
        self.delta_t = 3
        self.initial_mveh_glob_x = 340 # roughly middle of highway
        self.initial_time_step = 19
        self.time_step = 0
        self.vehicles = []
        self.initialize_env(state_arr)

    def initialize_env(self, state_arr):
        """Add vehicles to the scene and define which states have to be logged.
        """
        veh_ids = ['mveh' ,'yveh' ,'fveh', 'fadjveh']
        self.trace_log = {}

        for veh_id, veh_indx in zip(veh_ids, self.indxs.indxs):
            self.trace_log[veh_id] = {'glob_x':[], 'glob_y':[], 'lane_y':[],
                            'speed':[], 'act_long':[], 'act_lat':[]}
            if veh_id in ['mveh', 'fveh']:
                lane_id = 0
                glob_y = self.lane_width/2

            elif veh_id in ['yveh', 'fadjveh']:
                lane_id = 1
                glob_y = self.lane_width + self.lane_width/2

            if veh_id == 'mveh':
                glob_x =  self.initial_mveh_glob_x
                glob_y = self.lane_width/2 + state_arr[self.initial_time_step, veh_indx['pc']]

            elif veh_id == 'yveh':
                glob_x =  self.initial_mveh_glob_x - state_arr[self.initial_time_step, veh_indx['dx']]
            else:
                glob_x = self.initial_mveh_glob_x + state_arr[self.initial_time_step, veh_indx['dx']]

            speed = state_arr[self.initial_time_step, veh_indx['vel']]

            vehicle = HumanVehicle(veh_id, lane_id, glob_x, glob_y, speed)
            vehicle.act_long = state_arr[self.initial_time_step, veh_indx['act_long']]
            vehicle.act_lat = state_arr[self.initial_time_step, veh_indx['act_lat']]
            vehicle.actions = state_arr[self.initial_time_step:, veh_indx['act_long']:veh_indx['act_lat']+1]

            if veh_id == 'mveh':
                # mveh and caeveh share the same inital state
                vehicle.lane_y = state_arr[self.initial_time_step, veh_indx['pc']]
                self.caeveh = CAEVehicle('caeveh', lane_id, glob_x, glob_y, speed)
                self.caeveh.true_actions = vehicle.actions
                self.caeveh.act_long = vehicle.act_long
                self.caeveh.act_lat = vehicle.act_lat
                self.caeveh.lane_y = vehicle.lane_y
                self.caeveh.obs_history[0, :, :] = state_arr[:self.initial_time_step+1, :]
                self.vehicles.append(self.caeveh)
                self.trace_log['caeveh'] = {'glob_x':[], 'glob_y':[], 'lane_y':[],
                                'speed':[], 'act_long':[], 'act_lat':[]}

            self.vehicles.append(vehicle)

            caeveh_neighbours = {}
            for vehicle in self.vehicles:
                if vehicle.id != 'mveh' and vehicle.id != 'caeveh':
                    caeveh_neighbours[vehicle.id] = vehicle
            self.caeveh.neighbours = caeveh_neighbours

    def log_states(self, vehicle):
        for state_name in ['glob_x', 'glob_y', 'lane_y', 'speed', 'act_long', 'act_lat']:
            self.trace_log[vehicle.id][state_name].append(getattr(vehicle, state_name))

    def get_joint_action(self):
        joint_action = []
        for vehicle in self.vehicles:
            actions = vehicle.act(self.time_step)
            joint_action.append(actions)
        return joint_action

    def step(self):
        joint_action = self.get_joint_action()
        for t in range(self.delta_t):
            for vehicle, action in zip(self.vehicles, joint_action):
                vehicle.step(action[t, :])
                self.log_states(vehicle)
            obs_t = self.caeveh.neur_observe()
            self.caeveh.update_obs_history(obs_t)
            self.time_step += 1
