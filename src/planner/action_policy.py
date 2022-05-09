import numpy as np
import json
from importlib import reload
import os
import pickle
import tensorflow as tf
import dill
from collections import deque
from scipy.interpolate import CubicSpline
import time
from planner.state_indexs import StateIndxs
import sys
sys.path.insert(0, './src')
from evaluation.eval_data_obj import EvalDataObj

class Policy():
    STEP_SIZE = 0.1
    def __init__(self):
        self.discount_factor = 0.9
        self.delta_t = 3
        self.GAMMA = np.power(self.discount_factor, np.array(range(0,21)))
        self.pred_h = 20 # steps with 0.1 step size
        self.indxs = StateIndxs()

    def load_model(self, config, epoch):
        data_configs_path = './src/datasets/preprocessed/'
        exp_dir = './src/models/experiments/'+config['model_name']
        exp_path = f'{exp_dir}/model_epo{epoch}'

        with open('./src/datasets/'+'action_scaler', 'rb') as f:
            self.action_scaler = pickle.load(f)
        with open('./src/datasets/'+'state_scaler', 'rb') as f:
            self.state_scaler = pickle.load(f)

        self.step_size = config['data_config']['step_size']
        self.pred_step_n = np.ceil(self.pred_h/self.step_size).astype('int')

        from models.core import cae
        reload(cae)
        from models.core.cae import CAE
        self.model = CAE(config, model_use='inference')
        self.model.load_weights(exp_path).expect_partial()
        self.model.dec_model.steps_n = self.pred_step_n

    def inverse_transform_actions(self, _gen_actions):
        _gen_actions = np.concatenate(_gen_actions, axis=-1)
        _gen_actions.shape = (self.traj_n*(self.pred_step_n+1), 8)
        _gen_actions = self.action_scaler.inverse_transform(_gen_actions)
        _gen_actions.shape = (self.traj_n, self.pred_step_n+1, 8)
        gen_actions = [_gen_actions[:, :, n:n+2] for n in range(8)[::2] ]
        return gen_actions

    def cae_inference(self, inputs):
        """
        Uses CAE model in inference mode to generate an action sequence.
        Returns:
         - unscaled action seq
         - gmm_m (to be used for query of action likelihoods)
        """
        states, conds = inputs
        enc_state = self.model.enc_model(states)
        _gen_actions, gmm_m = self.model.dec_model([conds, enc_state])
        return _gen_actions, gmm_m

    def gen_action_seq(self, _gen_actions, conds):
        """ Inputs:
            - unscaled action seq
            Returns:
            - scaled action seq with first step conditional inserted in the sequence
        """
        gen_actions = [_act.numpy() for _act in _gen_actions]
        # t0 action is the conditional action
        gen_actions = []
        for cond, gen_action in zip(conds, _gen_actions):
            gen_action = np.insert(gen_action, 0, cond[:, 0, :], axis=1)
            gen_actions.append(gen_action)

        gen_actions = self.inverse_transform_actions(gen_actions)
        return gen_actions

    def get_boundary_condition(self, trace_history):
        """
        This is to ensure smooth transitions from one action to the next
        when using spline-based interpolation.
        """
        bc_ders = []
        for indx_act in self.indxs.indx_acts:
            bc_der = (trace_history[:, -1, indx_act[0]:indx_act[1]+1]-\
                    trace_history[:, -2, indx_act[0]:indx_act[1]+1])/self.STEP_SIZE
            bc_ders.append(bc_der)
        return bc_ders

    def get_pred_vehicle_plans(self, gen_actions, bc_ders):
        """Spline interpolation to turn action sequences into continuous plans
            for the vehicles of interest (including the agent).
        """
        if self.step_size == 1:
            # no need for spline-based interpolation
            gen_actions = [a[:, :self.pred_h+1, :]  for a in gen_actions]
            return gen_actions

        time_coarse = np.linspace(0, self.STEP_SIZE*self.step_size*self.pred_step_n, self.pred_step_n+1)
        time_fine = np.arange(0, time_coarse[-1]+0.05, self.STEP_SIZE)

        vehicle_plans = [] # action plans for all the vehicles
        for gen_action, bc_der in zip(gen_actions, bc_ders):
            f = CubicSpline(time_coarse, gen_action[:, :, :],
                                bc_type=((1, bc_der), (2, np.zeros([self.traj_n, 2]))),
                                axis=1)
            plans = f(time_fine)
            vehicle_plans.append(plans[:, :self.pred_h+1, :])
        return vehicle_plans

    def scale_state(self, state_t):
        state_t = state_t.copy()
        state_t[:, :, :-4] = (state_t[:, :, :-4]-\
                              self.state_scaler.mean_)/self.state_scaler.var_**0.5
        return state_t

    def normalize(self, seq):
        return seq/seq.max()

    def get_plan_likelihoods(self, _gen_actions, gmm_m):
        action_likelihoods = gmm_m.prob(_gen_actions[0]).numpy() # ensure actions are unscaled
        plan_likelihood = np.prod(action_likelihoods, axis=1)
        plan_likelihood = self.normalize(plan_likelihood)
        return plan_likelihood

    def plan_evaluation_func(self, plans_m, plan_likelihood, abs_distances):
        """
        Input:
        - plans_m: continuous, scaled merger plan options
        - plan_likelihood
        - abs_distances: distances between agent and other cars
        Return:
        - utility value for all the agent plans
        """
        jerk = (np.square(plans_m).sum(axis=-1)*self.GAMMA).sum(axis=-1)

        w = 4
        collision_cost = 0
        for abs_distance in abs_distances:
            mask = (abs_distance < 3.5).astype(int)
            collision_cost += (mask*(3.5-abs_distance)*700*self.GAMMA).sum(-1)

        plans_utility = -(jerk) + w*plan_likelihood - collision_cost
        return plans_utility

    def state_transition_function(self, state_0, action_plans, traj_n):
        """Used for assigning collision cost to agent plans
        Note: position of other cars are ego centric
        """
        state_0 = state_0.copy()
        # pred_trace: [dx_y, dx_f, dx_fadj, dy_y, dy_f, dy_fadj]
        pred_trace = np.zeros([traj_n, self.pred_h+1, 6])
        vel_m = state_0[:, self.indxs.indx_m['vel']]
        vel_y = state_0[:, self.indxs.indx_y['vel']]
        vel_f = state_0[:, self.indxs.indx_f['vel']]
        vel_fadj = state_0[:, self.indxs.indx_fadj['vel']]

        # relative vehicle distance in x axis
        pred_trace[:, 0, 0] = state_0[:, self.indxs.indx_y['dx']]
        pred_trace[:, 0, 1] = state_0[:, self.indxs.indx_f['dx']]
        pred_trace[:, 0, 2] = state_0[:, self.indxs.indx_fadj['dx']]

        # relative vehicle distance in y axis
        pred_trace[:, 0, 3] = 4 # lane width is roughly 4m
        pred_trace[:, 0, 4] = state_0[:, self.indxs.indx_m['pc']] #
        pred_trace[:, 0, 5] = 4

        for step in range(self.pred_h):
            # update dx values
            pred_trace[:, step+1, 0] = \
                        (pred_trace[:, step, 0]) + (vel_m-vel_y)*self.STEP_SIZE
            pred_trace[:, step+1, 1] = \
                        (pred_trace[:, step, 1]) + (vel_f-vel_m)*self.STEP_SIZE
            pred_trace[:, step+1, 2] = \
                        (pred_trace[:, step, 2]) + (vel_fadj-vel_m)*self.STEP_SIZE

            # update lateral position values
            act_lat_m = action_plans[0][:, step, 1]
            act_lat_y = action_plans[1][:, step, 1]
            act_lat_f = action_plans[2][:, step, 1]
            act_lat_fadj = action_plans[3][:, step, 1]

            dlatv = act_lat_y - act_lat_m
            pred_trace[:, step+1, 3] = pred_trace[:, step, 3] + dlatv*self.STEP_SIZE

            dlatv = act_lat_m - act_lat_f
            pred_trace[:, step+1, 4] = pred_trace[:, step, 4] + dlatv*self.STEP_SIZE

            dlatv = act_lat_fadj - act_lat_m
            pred_trace[:, step+1, 5] = pred_trace[:, step, 5] + dlatv*self.STEP_SIZE

            vel_m += action_plans[0][:, step, 0]*self.STEP_SIZE
            vel_y += action_plans[1][:, step, 0]*self.STEP_SIZE
            vel_f += action_plans[2][:, step, 0]*self.STEP_SIZE
            vel_fadj += action_plans[3][:, step, 0]*self.STEP_SIZE

        dis_y = (pred_trace[:, :, 0]**2 + pred_trace[:, :, 3]**2)**0.5
        dis_f = (pred_trace[:, :, 1]**2 + pred_trace[:, :, 4]**2)**0.5
        dis_fadj = (pred_trace[:, :, 2]**2 + pred_trace[:, :, 5]**2)**0.5
        return [dis_y, dis_f, dis_fadj]

    def mpc(self, trace_history, time_step):
        trace_history = np.repeat(\
                trace_history[:, :, :], self.traj_n, axis=0)

        states_i = self.scale_state(trace_history)
        conds_i = []
        for indx_act in self.indxs.indx_acts:
            conds_i.append(states_i[:, -1:, indx_act[0]:indx_act[1]+1])

        _gen_actions, gmm_m = self.cae_inference([states_i, conds_i])

        gen_actions = self.gen_action_seq(\
                            _gen_actions, conds_i)

        bc_ders = self.get_boundary_condition(trace_history)
        action_plans = self.get_pred_vehicle_plans(gen_actions, bc_ders)

        abs_distances = self.state_transition_function(trace_history[:, -1, :],\
                                                  action_plans,
                                                  self.traj_n)
        plan_likelihood = self.get_plan_likelihoods(_gen_actions, gmm_m)

        plans_utility = self.plan_evaluation_func(action_plans[0],
                                                 plan_likelihood,
                                                 abs_distances)

        best_plan = action_plans[0][np.argmax(plans_utility), :, :]

        if time_step == 0:
            return best_plan[:self.delta_t, :]
        else:
            return best_plan[1:self.delta_t+1, :]
