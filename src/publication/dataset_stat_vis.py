import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


m_col = ['episode_id', 'id', 'frm', 'vel', 'pc', 'lc_type', 'act_long_p',
                                            'act_lat_p', 'act_long', 'act_lat']

o_col = ['episode_id', 'id', 'frm', 'exists', 'vel', 'dx', 'act_long_p',
                                            'act_lat_p', 'act_long', 'act_lat']

m_df_lk = pd.read_csv('./src/datasets/m_df_lk.txt', delimiter=' ',
                                                        header=None, names=m_col)

m_df_lc = pd.read_csv('./src/datasets/m_df.txt', delimiter=' ',
                                                        header=None, names=m_col)
y_df_lc = pd.read_csv('./src/datasets/y_df.txt', delimiter=' ',
                                                        header=None, names=o_col)

spec_col = ['episode_id', 'scenario', 'lc_frm', 'm_id', 'y_id', 'fadj_id', 'f_id',
       'frm_n']
spec = pd.read_csv('./src/datasets/episode_spec.txt', delimiter=' ',
                                                        header=None, names=spec_col)


# %%
""" plot setup
"""
# plt.style.use('ieee')
plt.rcParams["font.family"] = "Times New Roman"
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# %%
""" Plot xy trajectories of vehicles. Use this to showcase what we are try to
learn to immitate.
"""

def get_trajs(df_epis, traj_len, starting_y=0):
    long_speed = df_epis['vel'].values
    lat_speed = df_epis['act_lat'].values
    pos_xy = np.zeros([traj_len, 2])
    pos_xy[0, 1] = starting_y

    for i in range(1, traj_len):
        pos_xy[i, 0] = pos_xy[i-1, 0] + long_speed[i-1]*0.1
        pos_xy[i, 1] = pos_xy[i-1, 1] + lat_speed[i-1]*0.1

    return pos_xy



plt.figure(figsize=(10, 7))
line_width = 1

episodes = m_df_lk['episode_id'].unique()
max_epis_in_plot = 600
# max_epis_in_plot = 1
lk_count = 0

for epis in episodes:
    df_epis = m_df_lk.loc[m_df_lk['episode_id'] == epis]
    traj_len = df_epis.shape[0]
    if 100 > traj_len > 50:
        pos_xy = get_trajs(df_epis, traj_len)
        plt.plot(pos_xy[:, 0], pos_xy[:, 1], linewidth = line_width)
        lk_count += 1

        if lk_count == max_epis_in_plot:
            break


episodes = m_df_lc['episode_id'].unique()
lc_left_count = 0
lc_right_count = 0

for epis in episodes:
    df_epis = m_df_lc.loc[m_df_lc['episode_id'] == epis]
    lc_type = df_epis['lc_type'].iloc[0]
    traj_len = df_epis.shape[0]
    if traj_len > 50:
        if lc_type == 1 and lc_left_count < max_epis_in_plot:
            pos_xy = get_trajs(df_epis, traj_len)
            plt.plot(pos_xy[:, 0], pos_xy[:, 1], linewidth = line_width)
            lc_left_count += 1

        if lc_type == -1 and lc_right_count < max_epis_in_plot:
            pos_xy = get_trajs(df_epis, traj_len)
            plt.plot(pos_xy[:, 0], pos_xy[:, 1], linewidth = line_width)
            lc_right_count += 1

        if lc_left_count == lc_right_count == max_epis_in_plot:
            break

plt.ylim(-5.5, 5.5)
plt.xlim(-0.1, 180)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig("ngsim_trajs.png", dpi=500, bbox_inches='tight')

# %%
"""
To showcase lane change extraction
"""
col = ['id','frm','scenario','lane_id',
                                'bool_r','bool_l','pc','vel',
                                'a_long','act_lat','a_lat','e_class',
                                'ff_id','ff_long','ff_lat','ff_v',
                                'fl_id','fl_long','fl_lat','fl_v',
                                'bl_id','bl_long','bl_lat','bl_v',
                                'fr_id','fr_long','fr_lat','fr_v',
                                'br_id','br_long','br_lat','br_v',
                                'bb_id','bb_long','bb_lat','bb_v']

col_drop = ['bool_r','bool_l','a_lat','a_long',
                    'fr_id','fr_long','fr_lat','fr_v',
                    'fl_id','fl_long','fl_lat','fl_v']

feature_set = pd.read_csv('./src/datasets/feature_set.txt', delimiter=' ',
                        header=None, names=col).drop(col_drop,axis=1)

# %%
# 76 i101_1 1519.0 525 526.0 518.0 522.0 66
df_epis = feature_set.loc[
                    (feature_set['scenario'] == 'i101_1') &
                    (feature_set['id'] == 525) &
                    (feature_set['frm'] >= 1519 - 80) &
                    (feature_set['frm'] <= 1519 + 100)].reset_index(drop = True)


# %%

plt.figure(figsize=(10, 7))
line_width = 3
episode_time = np.arange(df_epis.shape[0])*0.1 # seconds
pos_xy = get_trajs(df_epis, df_epis.shape[0], df_epis['pc'].min())
plt.plot([-1, 25], [0, 0], color='black', linestyle='--')
plt.plot([-1, 25], [df_epis['pc'].max(), df_epis['pc'].max()], color='red', linestyle='--')
plt.plot(episode_time, pos_xy[:, 1], linewidth = line_width, color='blue')

df_epis.where(df_epis['frm'] == 1519 + 66)
completion_index = df_epis[df_epis['frm'] == 1519 + 66].index[0]
completion_time = episode_time[completion_index]
completion_y = pos_xy[completion_index, 1]
plt.scatter(completion_time, completion_y, color='black', s=50)

# plt.ylim(-2.3, 3.5)
plt.xlim(0, 18)
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.savefig("lc_extraction_example.png", dpi=500, bbox_inches='tight')
# %%
"""
Visualising data distribution
I needed this to show that our dataset is sufficiently diverse.
"""
i101_episodes = spec.loc[
    (spec['scenario'] == 'i101_1') | \
    (spec['scenario'] == 'i101_2') | \
    (spec['scenario'] == 'i101_3')]['episode_id'].values

i80_episodes = spec.loc[
    (spec['scenario'] == 'i80_1') | \
    (spec['scenario'] == 'i80_2') | \
    (spec['scenario'] == 'i80_3')]['episode_id'].values


headway_i101_arr = y_df_lc[y_df_lc.episode_id.isin(i101_episodes)]['dx'].values
headway_i101_arr[np.isnan(headway_i101_arr)] = 70
headway_i101_arr[headway_i101_arr > 70] = 70
headway_i80_arr = y_df_lc[y_df_lc.episode_id.isin(i80_episodes)]['dx'].values
headway_i80_arr[np.isnan(headway_i80_arr)] = 70
headway_i80_arr[headway_i80_arr > 70] = 70
speed_i101_arr = m_df_lc[m_df_lc.episode_id.isin(i101_episodes)]['vel'].values
speed_i80_arr = m_df_lc[m_df_lc.episode_id.isin(i80_episodes)]['vel'].values

# %%
def set_up_ax(ax):
    ax.zaxis.axes._draw_grid = False
    tmp_planes = ax.zaxis._PLANES
    ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
                         tmp_planes[0], tmp_planes[1],
                         tmp_planes[4], tmp_planes[5])
    view_1 = (25, -135)
    view_2 = (25, -45)
    init_view = view_2
    ax.view_init(*init_view)
    ax.set_yticks([])
    ax.set_ylim([0., 2])
    ax.tick_params(axis='both', which='major', pad=10)
    ax.w_xaxis.set_pane_color((.66, .66, .66, 0.6))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))
    ax.view_init(25, -60)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Histogram count', rotation=90, labelpad=20)

def get_xz_poses(arr, bins):
    zs, xs = np.histogram(arr, bins=bins)
    dx = xs[1]-xs[0]
    x_poses = xs[1:] - dx/2
    return zs, xs, x_poses, dx

def sph2cart(r, theta, phi):
    '''spherical to cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def ravzip(*itr):
    '''flatten and zip arrays'''
    return zip(*map(np.ravel, itr))

def plot_bar(arr, ax, bins, orientation):
    assert orientation == 'rear' or orientation == 'front'
    zs, xs, x_poses, dx = get_xz_poses(arr, bins=bins)
    dy = 0.3
    front_rear_gap = 1

    if orientation == 'front':
        y_poses = np.zeros(x_poses.shape) + 0.2
        color = 'mediumseagreen'

    elif orientation == 'rear':
        y_poses = np.zeros(x_poses.shape) + 0.2 + dy + front_rear_gap
        color = 'lightblue'

    xyz = np.array(sph2cart(*sphview(ax_speed)), ndmin=3).T       #camera position in xyz
    zo = np.multiply([x_poses, y_poses, np.zeros_like(zs)], xyz).sum(0)  #"distance" of bars from camera
    for i, (x,y,dz,o) in enumerate(ravzip(x_poses, y_poses, zs, zo)):
        pl = ax.bar3d(x, y, 0, dx, dy, dz, color=color, edgecolor='black')
        pl._sort_zpos = o

fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2)
ax_headway = fig.add_subplot(1, 2, 1, projection='3d')
ax_headway.set_xlabel('Headway distance $\mathrm{(m)}$', rotation=0, labelpad=20)
ax_headway.set_xlim([0, 80])
ax_headway.plot([70, 70], [0, 2], [1, 1], linewidth=3, color='red')
set_up_ax(ax_headway)

ax_speed = fig.add_subplot(1, 2, 2, projection='3d')
ax_speed.set_xlabel('Long. speed $\mathrm{(ms^{-1}}$)', rotation=0, labelpad=20)
set_up_ax(ax_speed)

bins = 30
plot_bar(speed_i80_arr, ax_speed, bins, orientation='rear')
plot_bar(speed_i101_arr, ax_speed, bins, orientation='front')

plot_bar(headway_i80_arr, ax_headway, bins, orientation='rear')
plot_bar(headway_i101_arr, ax_headway, bins, orientation='front')

patch_green = patches.Rectangle(
        (0.1, 0.1),
        0.5,
        0.5,
        color='mediumseagreen')

patch_blue = patches.Rectangle(
        (0.1, 0.1),
        0.5,
        0.5,
        color='lightblue')

custom_lines = [patch_green,
                patch_blue]


# ax_headway.bar3d(70, -1, 0, 0.5, 3, 0.1, color='red')

ax_headway.legend(custom_lines, ['US 101 in Los Angeles', 'I-80 interstate in the San Francisco Bay Area'],
                  loc='lower center', bbox_to_anchor=(1, -0.3),
                fancybox=False, shadow=False, edgecolor=None, ncol=2, frameon=False)

plt.savefig("ngsim_state_distribution.png", dpi=500, bbox_inches='tight')
# %%
