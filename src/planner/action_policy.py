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
# from models.core.tf_models import cae_model
# reload(cae_model)
# from models.core.tf_models.cae_model import CAE
import sys
sys.path.insert(0, './src')

class Policy():
    def __init__(self, config):
        # self.data_obj = test_data.data_obj
        self.config = config
        self.discount_factor = 0.9
        self.gamma = np.power(self.discount_factor, np.array(range(0,21)))

    def load_model(self, epoch):
        exp_path = f'./src/models/experiments/{self.config["exp_id"]}/model_epo{epoch}'
        print(exp_path)
        print(os.getcwd())

        from models.core import cae
        reload(cae)
        from models.core.cae import CAE
        self.model = CAE(self.config, model_use='inference')
        self.model.load_weights(exp_path).expect_partial()

    def gen_action_seq(self, inputs, traj_n, steps_n):
        """
        Uses CAE model in inference mode to generate an action sequence.
        """
        state_history, conds = inputs

        if traj_n > 1:
            conds = [np.repeat(cond, traj_n, axis=0) for cond in conds]
            state_history = np.repeat(state_history, traj_n, axis=0)
            self.model.dec_model.steps_n = steps_n
            self.model.dec_model.traj_n = traj_n
        else:
            self.model.dec_model.steps_n = steps_n
            self.model.dec_model.traj_n = state_history.shape[0]


        enc_state = self.model.enc_model(state_history)


        _gen_actions = self.model.dec_model([conds, enc_state])
        _gen_actions = [_act.numpy() for _act in _gen_actions]

        # t0 action is the conditional action
        gen_actions = []
        for cond, gen_action in zip(conds, _gen_actions):
            gen_action = np.insert(gen_action, 0, cond[:, 0, :], axis=1)
            gen_actions.append(gen_action)
        return gen_actions

    def construct_policy(self, gen_actions, traj_n, steps_n):
        """
        Fit cubic splines to generated action sequences.
        TODO
        - scale acts
        - trip plans to length
        - add bc
        """
        time_coarse = np.linspace(0, step_size*(steps_n), steps_n+1)
        time_fine = np.arange(0, time_coarse[-1]+0.05, 0.1)

        vehicle_plans = [] # action plans for all the vehicles
        for gen_action in gen_actions:
            f = CubicSpline(time_coarse, gen_action[:, :, :], axis=1)
            coefs = np.stack(f.c, axis=2)
            plans = f(time_fine)
            vehicle_plans.append(plans)

        return vehicle_plans

    def get_actions(self, seq, bc_der, traj_n, pred_h):
        """
        :Return: unscaled action array for all cars
        """
        sampled_actions, _, _, prob_mlon, prob_mlat = self.get_cae_outputs(seq, traj_n, pred_h)

        act_mlon, act_mlat, act_y, act_f, act_fadj = sampled_actions
        st_seq, cond_seq = seq

        total_acts_count = traj_n*self.dec_model.steps_n
        veh_acts_count = 5 # 2 for merging, 1 for each of the other cars
        scaled_acts = np.zeros([total_acts_count, veh_acts_count])
        i = 0
        actions = [act_.numpy() for act_ in [act_mlon, act_mlat, act_y, act_f, act_fadj]]
        for act_ in actions:
            act_.shape = (total_acts_count)
            scaled_acts[:, i] = act_

            i += 1

        unscaled_acts = self.data_obj.action_scaler.inverse_transform(scaled_acts)
        unscaled_acts.shape = (traj_n, self.dec_model.steps_n, veh_acts_count)

        cond0 = [cond_seq[n][0, 0, :].tolist() for n in range(5)]
        cond0 = np.array([item for sublist in cond0 for item in sublist])
        cond0 = self.data_obj.action_scaler.inverse_transform(np.reshape(cond0, [1,-1]))
        cond0.shape = (1, 1, 5)
        cond0 = np.repeat(cond0, traj_n, axis=0)
        unscaled_acts = np.concatenate([cond0, unscaled_acts], axis=1)
        actions = self.construct_policy(unscaled_acts[:,0::self.skip_n,:], bc_der, traj_n, pred_h)

        return actions, prob_mlon, prob_mlat

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
