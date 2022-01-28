import numpy as np
import pickle
import sys


class Vehicle(object):
    STEP_SIZE = 0.1
    def __init__(self, id, lane_id, glob_x, glob_y, speed):
        self.id = id
        self.lane_id = lane_id
        self.glob_x = glob_x
        self.glob_y = glob_y
        self.speed = speed
        self.lane_y = 0
        self.lane_width = 3.75
        self.delta_t = 3

    def step(self, actions):
        self.act_long, self.act_lat = actions
        self.glob_x += self.speed * self.STEP_SIZE \
                                    + 0.5 * self.act_long * self.STEP_SIZE**2
        self.speed += self.act_long * self.STEP_SIZE
        self.glob_y += self.act_lat*self.STEP_SIZE
        self.lane_y += self.act_lat*self.STEP_SIZE

        if self.lane_y <= -self.lane_width/2:
            # just stepped into right lane
            self.lane_y += self.lane_width
        elif self.lane_y >= self.lane_width/2:
            # just stepped into left lane
            self.lane_y -= self.lane_width

    def act(self):
        raise NotImplementedError

class HumanVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, glob_y, speed):
        super().__init__(id, lane_id, glob_x, glob_y, speed)

    def act(self, time_step):
        return self.actions[time_step:time_step+self.delta_t, :]

class CAEVehicle(Vehicle):
    def __init__(self, id, lane_id, glob_x, glob_y, speed):
        super().__init__(id, lane_id, glob_x, glob_y, speed)
        self.samples_n = 1
        self.history_len = 20 # steps
        self.state_dim = 20
        self.obs_history = np.zeros([self.samples_n, self.history_len, self.state_dim])

    def update_obs_history(self, o_t):
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.obs_history[:, -1, :] = o_t

    def neur_observe(self):
        veh_ids = ['mveh' ,'yveh' ,'fveh', 'fadjveh']
        yveh = self.neighbours['yveh']
        fveh = self.neighbours['fveh']
        fadjveh = self.neighbours['fadjveh']
        obs_t = [self.speed, self.lane_y, self.act_long, self.act_lat,
                  yveh.speed, self.glob_x-yveh.glob_x, yveh.act_long, yveh.act_lat,
                  fveh.speed, fveh.glob_x-self.glob_x, fveh.act_long, fveh.act_lat,
                  fadjveh.speed, fadjveh.glob_x-self.glob_x, fadjveh.act_long, fadjveh.act_lat,
                  1, 1, 1, 1]


        return obs_t

    def act(self, time_step):
        return self.policy.mpc(self.obs_history, time_step, 10)
