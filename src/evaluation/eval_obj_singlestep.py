"""
Use this module for evaluating step-wise LSTM and MLP-based models.
"""
import sys
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVALMultiStep
from planner.state_indexs import StateIndxs
import numpy as np

class ForwardSimSingleStep():
    STEP_SIZE = 0.1
    def __init__(self):
        self.indxs = StateIndxs()

    def get_dx(self, vel_m, vel_o, veh_orientation):
        """
        veh_orientation is mering vehicle's orientation relative to others
        """
        if veh_orientation == 'front':
            dv = vel_m - vel_o
        else:
            dv = vel_o - vel_m

        dx = dv*self.STEP_SIZE
        return dx

    def step(self, state_t_i, state_t_ii, act_m):
        state_t_ii[:, self.indxs.indx_m['act_long']] = act_m[:, 0]
        state_t_ii[:, self.indxs.indx_m['act_lat']] = act_m[:, 1]

        next_vel = state_t_i[:, self.indxs.indx_m['vel']] + act_m[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indxs.indx_m['vel']] = next_vel

        next_pc = state_t_i[:, self.indxs.indx_m['pc']] + act_m[:, 1]*self.STEP_SIZE
        lc_left = next_pc > self.max_pc
        next_pc[lc_left] = self.min_pc
        lc_right = next_pc < self.min_pc
        next_pc[lc_right] = self.max_pc
        state_t_ii[:, self.indxs.indx_m['pc']] = next_pc

        vel_m = state_t_i[:, self.indxs.indx_m['vel']]
        vel_y = state_t_i[:, self.indxs.indx_y['vel']]
        vel_f = state_t_i[:, self.indxs.indx_f['vel']]
        vel_fadj = state_t_i[:, self.indxs.indx_fadj['vel']]

        next_dx = state_t_i[:, self.indxs.indx_y['dx']] + self.get_dx(vel_m, vel_y, 'front')
        state_t_ii[:, self.indxs.indx_y['dx']] = next_dx
        next_dx = state_t_i[:, self.indxs.indx_f['dx']] + self.get_dx(vel_m, vel_f, 'behind')
        state_t_ii[:, self.indxs.indx_f['dx']] = next_dx
        next_dx = state_t_i[:, self.indxs.indx_fadj['dx']] + self.get_dx(vel_m, vel_f, 'behind')
        state_t_ii[:, self.indxs.indx_fadj['dx']] = next_dx

class MCEVALSingleStep(MCEVALMultiStep):
    def __init__(self, config):
        super().__init__(config)
        self.fs = ForwardSimSingleStep()
        self.pred_h = 20 # steps with 0.1 step size

    def scale_state(self, state_t):
        state_t = state_t.copy()
        state_dims = state_t.shape
        if len(state_dims) == 2:
            state_t[:, :-4] = self.state_scaler.transform(state_t[:, :-4])

        elif len(state_dims) == 3:
            state_t.shape = (state_dims[0]*self.obs_n, state_dims[-1])
            state_t[:, :-4] = self.state_scaler.transform(state_t[:, :-4])
            state_t.shape = (state_dims[0], self.obs_n, state_dims[-1])

        return state_t

    def inverse_transform_actions(self, _gen_actions):
        _gen_actions = np.insert(_gen_actions, 2, np.zeros([6, 1]), axis=1)
        _gen_actions = self.action_scaler.inverse_transform(_gen_actions)
        return _gen_actions[:, :2]

    def get_predicted_trace(self, states, conds, true_trace):
        if self.model_type == 'MLP':
            pred_trace = np.repeat(\
                    true_trace[:, self.obs_n-1:, 2:], self.traces_n, axis=0)

            for step in range(self.pred_h):
                gmm_m = self.policy(self.scale_state(pred_trace[:, step, :]))
                act_m = self.inverse_transform_actions(gmm_m.sample().numpy())
                act_m = np.clip(act_m, -10, 10)
                assert not np.isnan(act_m).any(), 'There is a nan in sampled actions!'
                self.fs.step(pred_trace[:, step, :], pred_trace[:, step+1, :], act_m)

        elif self.model_type == 'LSTM':
            true_trace = np.repeat(true_trace[:, :, 2:], self.traces_n, axis=0)
            pred_trace = true_trace[:, self.obs_n-1:, :]
            state_history = self.scale_state(true_trace[:, :self.obs_n, :])

            for step in range(self.pred_h):
                gmm_m = self.policy(state_history)
                act_m = self.inverse_transform_actions(gmm_m.sample().numpy())
                act_m = np.clip(act_m, -10, 10)
                assert not np.isnan(act_m).any(), 'There is a nan in sampled actions!'
                self.fs.step(pred_trace[:, step, :], pred_trace[:, step+1, :], act_m)
                state_history[:, :-1, :] = state_history[:, 1:, :]
                state_history[:, -1, :] = self.scale_state(pred_trace[:, step+1, :])

        return pred_trace
