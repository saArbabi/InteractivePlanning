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
from src.planner.state_indexs import StateIndxs
import sys
sys.path.insert(0, './src')

class Policy():
    STEP_SIZE = 0.1
    def __init__(self):
        # self.data_obj = test_data.data_obj
        self.discount_factor = 0.9
        self.gamma = np.power(self.discount_factor, np.array(range(0,21)))
        self.pred_h = 21 # steps with 0.1 step size
        self.indxs = StateIndxs()

    def load_model(self, model_name):
        epoch = 20
        exp_dir = './src/models/experiments/'+model_name
        exp_path = f'{exp_dir}/model_epo{epoch}'
        with open(exp_dir+'/'+'config.json', 'rb') as handle:
            config = json.load(handle)

        self.step_size = config['data_config']['step_size']
        self.pred_step_n = np.ceil(self.pred_h/self.step_size).astype('int')

        from models.core import cae
        reload(cae)
        from models.core.cae import CAE
        self.model = CAE(config, model_use='inference')
        self.model.load_weights(exp_path).expect_partial()
        self.model.dec_model.steps_n = self.pred_step_n

    def inverse_transform_actions(self, _gen_actions, traj_n):
        _gen_actions = np.concatenate(_gen_actions, axis=-1)
        _gen_actions.shape = (traj_n*(self.pred_step_n+1), 8)
        _gen_actions = self.action_scaler.inverse_transform(_gen_actions)
        _gen_actions.shape = (traj_n, self.pred_step_n+1, 8)
        gen_actions = [_gen_actions[:, :, n:n+2] for n in range(8)[::2] ]
        return gen_actions

    def gen_action_seq(self, inputs, traj_n):
        """
        Uses CAE model in inference mode to generate an action sequence.
        """
        state_history, conds = inputs

        if traj_n > 1:
            conds = [np.repeat(cond, traj_n, axis=0) for cond in conds]
            state_history = np.repeat(state_history, traj_n, axis=0)
            self.model.dec_model.traj_n = traj_n
        else:
            self.model.dec_model.traj_n = state_history.shape[0]

        enc_state = self.model.enc_model(state_history)

        _gen_actions = self.model.dec_model([conds, enc_state])
        gen_actions = [_act.numpy() for _act in _gen_actions]
        # t0 action is the conditional action
        gen_actions = []
        for cond, gen_action in zip(conds, _gen_actions):
            gen_action = np.insert(gen_action, 0, cond[:, 0, :], axis=1)
            gen_actions.append(gen_action)

        gen_actions = self.inverse_transform_actions(gen_actions, traj_n)
        return gen_actions

    def get_boundary_condition(self, state_history):
        """
        This is to ensure smooth transitions from one action to the next.
        The bc is the first time derivative of a plan at current time step.
        Note: Time derivative is the same regardless of action scale.
        """
        bc_ders = []
        for indx_act in self.indxs.indx_acts:
            bc_der = (state_history[:, -1, indx_act]-\
                                    state_history[:, -2, indx_act])/self.STEP_SIZE

            bc_ders.append(bc_der)
        return bc_ders

    def construct_policy(self, gen_actions, state_history):
        if self.step_size == 1:
            return gen_actions

        bc_ders = self.get_boundary_condition(state_history)
        time_coarse = np.linspace(0, self.STEP_SIZE*self.step_size*self.pred_step_n, self.pred_step_n+1)
        time_fine = np.arange(0, time_coarse[-1]+0.05, self.STEP_SIZE)

        vehicle_plans = [] # action plans for all the vehicles
        for gen_action, bc_der in zip(gen_actions, bc_ders):
            f = CubicSpline(time_coarse, gen_action[:, :8, :],
                                bc_type=((1, bc_der), (2, np.zeros([20, 2]))),
                                axis=1)
            plans = f(time_fine)
            vehicle_plans.append(plans[:, :self.pred_h, :])

        return vehicle_plans

    def objective(self, actions, prob_mlon, prob_mlat):
        """To evaluate the plans, and select the best one
        """
        jerk = actions[:,:,0]**2
        jerk_norm = jerk/np.repeat([np.max(jerk, axis=0)], 2000, axis=0)
        likelihoods = np.prod(prob_mlon, axis=1).flatten()+np.prod(prob_mlat, axis=1).flatten()
        j = np.sum(jerk_norm*self.gamma, axis=1)
        j_weight = 1
        likelihood_weight = 3.5

        discounted_cost = -j_weight*j  + likelihood_weight*likelihoods/max(likelihoods)
        best_plan_indx = np.where(discounted_cost==max(discounted_cost))[0][0]
        return best_plan_indx

    def mpc(self, obs_history, action_conditional, bc):
        """bc is the boundary condition for spline fitting
        """
        actions, prob_mlon, prob_mlat = self.get_actions(\
                [obs_history, action_conditional], bc, traj_n=2000, pred_h=2)
        best_plan_indx = self.objective(actions, prob_mlon, prob_mlat)
        all_future_plans = actions[best_plan_indx, 1:self.replanning_rate+1, :]
        ego_plan = all_future_plans[:, 0:2]
        bc = (all_future_plans[-1]-all_future_plans[-2])*10
        return ego_plan, bc
