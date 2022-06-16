import numpy as np
import matplotlib.pyplot as plt
from planner.state_indexs import StateIndxs


class Viewer():
    def __init__(self, trace_log):
        self.indxs = StateIndxs()
        self.env_config = {
                            'lane_width': 4,
                            'lane_length': 300,
                            'lane_count': 3}
        self.trace_log = trace_log

    def set_up_traffic_intro_fig(self):
        self.traffic_intro_fig = plt.figure(figsize=(10, 3))
        self.traffic_intro_fig.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.3)
        self.speeds = self.traffic_intro_fig.add_subplot(111)
        self.speeds.set_xlabel(r'Time (s)')
        self.speeds.set_ylabel(r'Long. speed (m/s)')
        self.speeds.yaxis.set_ticks(range(0, 15, 2))
        self.speeds.set_ylim(7.5, 12.5)
        self.speeds.set_xlim(0, 9)


    def set_up_traffic_fig(self):
        self.traffic_fig = plt.figure(figsize=(10, 9))
        self.traffic_fig.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.4)
        self.scene_t1_ax = self.traffic_fig.add_subplot(311, facecolor='lightgrey')
        self.scene_t2_ax = self.traffic_fig.add_subplot(312, facecolor='lightgrey')
        self.scene_t3_ax = self.traffic_fig.add_subplot(313, facecolor='lightgrey')
        for ax in self.traffic_fig.axes:
            self.draw_road(ax)

    def set_up_profile_fig(self):
        self.state_profile_fig = plt.figure(figsize=(10, 9))
        self.state_profile_fig.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.4)
        self.speed_ax = self.state_profile_fig.add_subplot(311)
        self.act_long_ax = self.state_profile_fig.add_subplot(312)
        self.act_lat_ax = self.state_profile_fig.add_subplot(313)
        for ax in self.state_profile_fig.axes:
            ax.grid(alpha=0.6)
            ax.set_xlim(-0.2, 6)
            ax.set_xlabel(r'Time (s)')

    def draw_speeds(self, state_arr):
        traj_len = len(state_arr[:, self.indxs.indx_m['vel']])
        time_steps = np.linspace(0, traj_len*0.1, traj_len)

        speed_prof = state_arr[:, self.indxs.indx_m['vel']]
        self.speeds.plot(time_steps, speed_prof, color='orange', linewidth=2.5, linestyle='-')

        speed_prof = state_arr[:, self.indxs.indx_y['vel']]
        self.speeds.plot(time_steps, speed_prof, color='gold', linewidth=2.5, linestyle='--')

        speed_prof = state_arr[:, self.indxs.indx_f['vel']]
        self.speeds.plot(time_steps, speed_prof, color='green', linewidth=2.5, linestyle='-.')

        speed_prof = state_arr[:, self.indxs.indx_fadj['vel']]
        self.speeds.plot(time_steps, speed_prof, color='royalblue', linewidth=2.5, linestyle=':')

        self.speeds.legend(['$e$', '$v_1$', '$v_2$', '$v_3$'], ncol=1, edgecolor='black')
        self.speeds.plot([time_steps[19], time_steps[19]], [5, 13], color='black',
                                                                linestyle='--')


    def draw_road(self, ax):
        lane_cor = self.env_config['lane_width']*self.env_config['lane_count']
        ax.hlines(0, 0, self.env_config['lane_length'], colors='k', linestyles='solid')
        ax.hlines(lane_cor, 0, self.env_config['lane_length'],
                            colors='k', linestyles='solid')

        if self.env_config['lane_count'] > 1:
            lane_cor = self.env_config['lane_width']
            for lane in range(self.env_config['lane_count']-1):
                ax.hlines(lane_cor, 330, 430,
                                colors='white', linestyles='--', linewidth=4)
                lane_cor += self.env_config['lane_width']
        ax.set_xlim(330, 430)
        ax.set_ylim(0, 12.1)
        ax.yaxis.set_ticks(range(0, 13, 4))
        ax.set_xlabel(r' Longitudinal position (m)')
        ax.set_ylabel(r'Lateral position (m)')

    def draw_state_profiles(self):
        time_steps = np.linspace(0, 5.9, 60)
        traj_len = len(self.trace_log['mveh']['speed'])
        lw = 2.5

        self.speed_ax.plot(
            time_steps[:traj_len], self.trace_log['mveh']['speed'], color='red', linestyle='--', linewidth=lw)
        self.speed_ax.plot(
            time_steps[:traj_len], self.trace_log['caeveh']['speed'], color='green', linestyle='-', linewidth=lw)
        self.speed_ax.set_ylabel(r'Long. speed (m/s)')
        self.speed_ax.yaxis.set_ticks(np.arange(10., 13, 1))
        self.speed_ax.set_ylim(10, 12.5)

        self.act_long_ax.plot(
            time_steps[:traj_len], self.trace_log['mveh']['act_long'], color='red', linestyle='--', linewidth=lw)
        self.act_long_ax.plot(
            time_steps[:traj_len], self.trace_log['caeveh']['act_long'], color='green', linestyle='-', linewidth=lw)
        self.act_long_ax.yaxis.set_ticks(np.arange(-2, 2.1, 1))
        self.act_long_ax.set_ylabel(r'Long. Accel. $\mathdefault{(m/s^2)}$')

        self.act_lat_ax.plot(
            time_steps[:traj_len], self.trace_log['mveh']['act_lat'], color='red', linestyle='--', linewidth=lw)
        self.act_lat_ax.plot(
            time_steps[:traj_len], self.trace_log['caeveh']['act_lat'], color='green', linestyle='-', linewidth=lw)
        self.act_lat_ax.yaxis.set_ticks(np.arange(-2, 2.1, 1))
        self.act_lat_ax.set_ylabel(r'Lateral speed (m/s)')

    def draw_xy_profile(self):
        traj_len = len(self.trace_log['mveh']['speed'])
        snap_count = 3 # total number of snapshots of a given trajectory
        snap_interval = int(np.floor(traj_len/snap_count))

        chunk = 0
        for snap_i in range(snap_count):
            true_x = self.trace_log['mveh']['glob_x'][:chunk+snap_interval]
            true_y = self.trace_log['mveh']['glob_y'][:chunk+snap_interval]
            self.traffic_fig.axes[snap_i].plot(\
                    true_x, true_y, color='red', linestyle='--', linewidth=2.5)

            true_x = self.trace_log['yveh']['glob_x'][chunk+snap_interval-1]
            true_y = self.trace_log['yveh']['glob_y'][chunk+snap_interval-1]
            self.traffic_fig.axes[snap_i].scatter(true_x, true_y, color='black', s=5)

            true_x = self.trace_log['fveh']['glob_x'][chunk+snap_interval-1]
            true_y = self.trace_log['fveh']['glob_y'][chunk+snap_interval-1]
            self.traffic_fig.axes[snap_i].scatter(true_x, true_y, color='green', s=5)

            true_x = self.trace_log['fadjveh']['glob_x'][chunk+snap_interval-1]
            true_y = self.trace_log['fadjveh']['glob_y'][chunk+snap_interval-1]
            self.traffic_fig.axes[snap_i].scatter(true_x, true_y, color='blue', s=5)

            pred_x = self.trace_log['caeveh']['glob_x'][:chunk+snap_interval]
            pred_y = self.trace_log['caeveh']['glob_y'][:chunk+snap_interval]
            self.traffic_fig.axes[snap_i].plot(\
                            pred_x, pred_y, color='green', linewidth=2.5)

            chunk += snap_interval



    def draw_plots(self):
        self.draw_state_profiles()
        self.draw_xy_profile()



        # self.draw_road(self.scene_t1_ax)
        # self.draw_road(self.scene_t2_ax)
        # self.draw_road(self.scene_t3_ax)
        # self.draw_xy_profile([self.scene_t1_ax,
        #                         self.scene_t2_ax,
        #                         self.scene_t3_ax],
        #                         vehicles)

        # pad_inches = 0)
        # plt.close()
        # self.draw_v_profile(self.speed_ax, vehicles)
        # self.draw_along_profile(self.act_long_ax, vehicles)
        # self.draw_alat_profile(self.act_lat_ax, vehicles)
        # pad_inches = 0)
        # plt.close()
                # plt.show()
