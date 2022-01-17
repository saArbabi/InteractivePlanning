import numpy as np

class ForwardSim():
    STEP_SIZE = 0.1
    def __init__(self):
        self.set_stateIndex()

    def set_stateIndex(self):
        self.indx_m = {}
        self.indx_y = {}
        self.indx_f = {}
        self.indx_fadj = {}
        i = 0
        for name in ['vel', 'pc', 'act_long','act_lat']:
            self.indx_m[name] = i
            i += 1
        for name in ['vel', 'dx', 'act_long','act_lat']:
            self.indx_y[name] = i
            i += 1
        for name in ['vel', 'dx', 'act_long','act_lat']:
            self.indx_f[name] = i
            i += 1
        for name in ['vel', 'dx', 'act_long','act_lat']:
            self.indx_fadj[name] = i
            i += 1

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

    def step(self, state_t_i, veh_actions):
        act_m, act_y, act_f, act_fadj = veh_actions
        state_t_ii = state_t_i.copy()
        state_t_ii[:, self.indx_m['vel']] += act_m[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indx_y['vel']] += act_y[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indx_f['vel']] += act_f[:, 0]*self.STEP_SIZE
        state_t_ii[:, self.indx_fadj['vel']] += act_fadj[:, 0]*self.STEP_SIZE

        state_t_ii[:, self.indx_m['pc']] += act_m[:, 1]*self.STEP_SIZE
        lc_left = state_t_ii[:, self.indx_m['pc']] > self.max_pc
        state_t_ii[lc_left, self.indx_m['pc']] = self.min_pc
        lc_right = state_t_ii[:, self.indx_m['pc']] < self.min_pc
        state_t_ii[lc_right, self.indx_m['pc']] = self.max_pc

        vel_m = state_t_i[:, self.indx_m['vel']]
        vel_y = state_t_i[:, self.indx_y['vel']]
        vel_f = state_t_i[:, self.indx_f['vel']]
        vel_fadj = state_t_i[:, self.indx_fadj['vel']]
        state_t_ii[:, self.indx_y['dx']] += self.get_dx(vel_m, vel_y, 'front')
        state_t_ii[:, self.indx_f['dx']] += self.get_dx(vel_m, vel_f, 'behind')
        state_t_ii[:, self.indx_fadj['dx']] += self.get_dx(vel_m, vel_fadj, 'behind')

        state_t_ii[:, self.indx_m['act_long']] = act_m[:, 0]
        state_t_ii[:, self.indx_m['act_lat']] = act_m[:, 1]

        state_t_ii[:, self.indx_y['act_long']] = act_y[:, 0]
        state_t_ii[:, self.indx_y['act_lat']] = act_y[:, 1]

        state_t_ii[:, self.indx_f['act_long']] = act_f[:, 0]
        state_t_ii[:, self.indx_f['act_lat']] = act_f[:, 1]

        state_t_ii[:, self.indx_fadj['act_long']] = act_fadj[:, 0]
        state_t_ii[:, self.indx_fadj['act_lat']] = act_fadj[:, 1]

        return state_t_ii

    def forward_sim(self, state_0, action_plans):
        # state_0 is initial traffic state
        steps_n = action_plans[0].shape[1]
        trajs_n = action_plans[0].shape[0]
        states_n = state_0.shape[-1]

        state_trace = np.zeros([trajs_n, steps_n+1, states_n])
        state_trace[:, 0, :] = state_0
        for step in range(steps_n):
            veh_actions = [plan[:, step, :] for plan in action_plans]
            state_trace[:, step+1, :] = self.step(state_trace[:, step, :], veh_actions)

        return state_trace
