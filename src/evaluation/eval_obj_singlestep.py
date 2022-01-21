import sys
sys.path.insert(0, './src')
from evaluation.eval_obj import MCEVAL

class ForwardSimSingleStep():
    STEP_SIZE = 0.1
    def __init__(self):
        self.indxs = StateIndxs()
        self.pred_h = 20 # steps with 0.1 step size

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

    def update_veh_actions(self, state_t, act_m):
        """
        In state state_t, the vehicles take actions veh_actions.
        """
        state_t[:, self.indxs.indx_m['act_long']] = act_m[:, 0]
        state_t[:, self.indxs.indx_m['act_lat']] = act_m[:, 1]

    def step(self, state_t_i, act_m):
        state_t_ii = state_t_i.copy()
        state_t_ii[:, self.indxs.indx_m['vel']] += act_m[:, 0]*self.STEP_SIZE

        state_t_ii[:, self.indxs.indx_m['pc']] += act_m[:, 1]*self.STEP_SIZE
        lc_left = state_t_ii[:, self.indxs.indx_m['pc']] > self.max_pc
        state_t_ii[lc_left, self.indxs.indx_m['pc']] = self.min_pc
        lc_right = state_t_ii[:, self.indxs.indx_m['pc']] < self.min_pc
        state_t_ii[lc_right, self.indxs.indx_m['pc']] = self.max_pc

        vel_m = state_t_i[:, self.indxs.indx_m['vel']]
        vel_y = state_t_i[:, self.indxs.indx_y['vel']]
        vel_f = state_t_i[:, self.indxs.indx_f['vel']]
        vel_fadj = state_t_i[:, self.indxs.indx_fadj['vel']]
        state_t_ii[:, self.indxs.indx_y['dx']] += self.get_dx(vel_m, vel_y, 'front')
        state_t_ii[:, self.indxs.indx_f['dx']] += self.get_dx(vel_m, vel_f, 'behind')
        state_t_ii[:, self.indxs.indx_fadj['dx']] += self.get_dx(vel_m, vel_fadj, 'behind')

        return state_t_ii


class MCEVALSingleStep(MCEVAL):
    def __init__(self, val_run_name=None):
        super().__init__(val_run_name)
        self.fs = ForwardSimSingleStep()

    def get_predicted_trace(self, states, conds, true_trace):
        true_trace_history = np.repeat(\
                true_trace[:, :self.obs_n-1, 2:], self.traces_n, axis=0)
        state0 = true_trace_history[:, -1, :]

        trajs_n = action_plans[0].shape[0]
        pred_trace = np.zeros([trajs_n, self.pred_h+1, state_0.shape[-1]])
        state_t_i = state_0

        for step in range(self.pred_h):

            ego_actions = self.policy.gen_action_seq(\
                                [states, conds], traj_n=self.traces_n)


            veh_actions = [plan[:, step, :] for plan in action_plans]
            self.update_veh_actions(state_t_i, veh_actions)
            pred_trace[:, step, :] = state_t_i

            state_t_ii = self.step(state_t_i, veh_actions)
            pred_trace[:, step+1, :] = state_t_ii
            state_t_i = state_t_ii

        veh_actions = [plan[:, -1, :] for plan in action_plans] # last state actions
        self.update_veh_actions(state_t_i, veh_actions)
        pred_trace[:, -1, :] = state_t_i

        return pred_trace
