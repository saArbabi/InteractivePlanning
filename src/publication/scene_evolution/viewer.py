import numpy as np
import matplotlib.pyplot as plt
class Viewer():
    def __init__(self, trace_log):
        # plt.rcParams.update({'font.size': 14})
        self.env_config = {
                            'lane_width': 4,
                            'lane_length': 300,
                            'lane_count': 3}
        self.trace_log = trace_log
        self.set_up_figures()

    def set_up_figures(self):
        self.traffic_fig = plt.figure(figsize=(10, 9))
        self.traffic_fig.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.3)
        self.scene_t1_ax = self.traffic_fig.add_subplot(311, facecolor='lightgrey')
        self.scene_t2_ax = self.traffic_fig.add_subplot(312, facecolor='lightgrey')
        self.scene_t3_ax = self.traffic_fig.add_subplot(313, facecolor='lightgrey')
        for ax in self.traffic_fig.axes:
            self.draw_road(ax)



        self.state_profile_fig = plt.figure(figsize=(10, 9))
        self.state_profile_fig.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.3)
        self.speed_ax = self.state_profile_fig.add_subplot(311)
        self.act_long_ax = self.state_profile_fig.add_subplot(312)
        self.act_lat_ax = self.state_profile_fig.add_subplot(313)
        for ax in self.state_profile_fig.axes:
            ax.grid(alpha=0.6)
            ax.set_xlim(0, 6)
            ax.set_xlabel('Time ($s$)')

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
        ax.set_xlabel(' Longitudinal position ($m$)')
        ax.set_ylabel('Lateral position ($m$)')

    def draw_state_profiles(self):
        time_steps = np.linspace(0, 5.9, 60)
        traj_len = len(self.trace_log['mveh']['speed'])

        self.speed_ax.plot(time_steps[:traj_len], self.trace_log['mveh']['speed'], color='red')
        self.speed_ax.plot(time_steps[:traj_len], self.trace_log['caeveh']['speed'], color='green')
        self.speed_ax.set_ylabel('Long. speed ($ms^{-1}$)')
        self.speed_ax.yaxis.set_ticks(np.arange(10.5, 13, 0.5))

        self.act_long_ax.plot(time_steps[:traj_len], self.trace_log['mveh']['act_long'], color='red')
        self.act_long_ax.plot(time_steps[:traj_len], self.trace_log['caeveh']['act_long'], color='green')
        self.act_long_ax.yaxis.set_ticks(np.arange(-2, 2.1, 1))
        self.act_long_ax.set_ylabel('Long. acceleration ($ms^{-2}$)')

        self.act_lat_ax.plot(time_steps[:traj_len], self.trace_log['mveh']['act_lat'], color='red')
        self.act_lat_ax.plot(time_steps[:traj_len], self.trace_log['caeveh']['act_lat'], color='green')
        self.act_lat_ax.yaxis.set_ticks(np.arange(-2, 2.1, 1))
        self.act_lat_ax.set_ylabel('Lateral speed ($ms^{-1}$)')

    def draw_xy_profile(self):
        traj_len = len(self.trace_log['mveh']['speed'])
        snap_count = 3 # total number of snapshots of a given trajectory
        snap_interval = int(np.floor(traj_len/snap_count))

        chunk = 0
        for snap_i in range(snap_count):
            true_x = self.trace_log['mveh']['glob_x'][:chunk+snap_interval]
            true_y = self.trace_log['mveh']['glob_y'][:chunk+snap_interval]
            self.traffic_fig.axes[snap_i].plot(\
                            true_x, true_y, color='red', linestyle='--')

            true_x = self.trace_log['yveh']['glob_x'][chunk+snap_interval-1]
            true_y = self.trace_log['yveh']['glob_y'][chunk+snap_interval-1]
            self.traffic_fig.axes[snap_i].scatter(true_x, true_y, color='yellow')

            true_x = self.trace_log['fveh']['glob_x'][chunk+snap_interval-1]
            true_y = self.trace_log['fveh']['glob_y'][chunk+snap_interval-1]
            self.traffic_fig.axes[snap_i].scatter(true_x, true_y, color='green')

            true_x = self.trace_log['fadjveh']['glob_x'][chunk+snap_interval-1]
            true_y = self.trace_log['fadjveh']['glob_y'][chunk+snap_interval-1]
            self.traffic_fig.axes[snap_i].scatter(true_x, true_y, color='blue')

            pred_x = self.trace_log['caeveh']['glob_x'][:chunk+snap_interval]
            pred_y = self.trace_log['caeveh']['glob_y'][:chunk+snap_interval]
            self.traffic_fig.axes[snap_i].plot(pred_x, pred_y, color='green')

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
        # self.traffic_fig.savefig("env_evolution.png", dpi=500, bbox_inches = 'tight',
        # pad_inches = 0)
        # plt.close()
        # self.draw_v_profile(self.speed_ax, vehicles)
        # self.draw_along_profile(self.act_long_ax, vehicles)
        # self.draw_alat_profile(self.act_lat_ax, vehicles)
        # self.state_profile_fig.savefig("env_profiles.png", dpi=500, bbox_inches = 'tight',
        # pad_inches = 0)
        # plt.close()
                # plt.show()
