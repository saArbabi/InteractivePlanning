import numpy as np
from planner.state_indexs import StateIndxs

class ForwardSim():
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

    def update_veh_actions(self, state_t, veh_actions):
        """
        In state state_t, the vehicles take actions veh_actions.
        """
        act_m, act_y, act_f, act_fadj = veh_actions
        state_t[:, self.indxs.indx_m['act_long']] = act_m[:, 0]
        state_t[:, self.indxs.indx_m['act_lat']] = act_m[:, 1]

        state_t[:, self.indxs.indx_y['act_long']] = act_y[:, 0]
        state_t[:, self.indxs.indx_y['act_lat']] = act_y[:, 1]

        state_t[:, self.indxs.indx_f['act_long']] = act_f[:, 0]
        state_t[:, self.indxs.indx_f['act_lat']] = act_f[:, 1]

        state_t[:, self.indxs.indx_fadj['act_long']] = act_fadj[:, 0]
        state_t[:, self.indxs.indx_fadj['act_lat']] = act_fadj[:, 1]

    def step(self, state_t_i, veh_actions):
        act_m, act_y, act_f, act_fadj = veh_actions
        state_t_ii = state_t_i.copy()
        state_t_ii[:, self.indxs.indx_m['vel']] += act_m[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indxs.indx_y['vel']] += act_y[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indxs.indx_f['vel']] += act_f[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indxs.indx_fadj['vel']] += act_fadj[:, 0]*self.STEP_SIZE

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

    def forward_sim(self, state_0, action_plans):
        # state_0 is initial traffic state
        # s_0 >> a_0 >> s_1 >> a_1 >> s_2 >> a_2 >> s_3...
        trajs_n = action_plans[0].shape[0]
        pred_trace = np.zeros([trajs_n, self.pred_h+1, state_0.shape[-1]])
        state_t_i = state_0

        for step in range(self.pred_h):
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
