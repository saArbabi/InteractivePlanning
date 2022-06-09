import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


m_col = ['episode_id', 'id', 'frm', 'vel', 'pc', 'lc_type', 'act_long_p',
                                            'act_lat_p', 'act_long', 'act_lat']



m_df_lk = pd.read_csv('./src/datasets/m_df_lk.txt', delimiter=' ',
                                                        header=None, names=m_col)

m_df_lc = pd.read_csv('./src/datasets/m_df.txt', delimiter=' ',
                                                        header=None, names=m_col)


# %%
""" plot setup
"""
# plt.style.use('ieee')
plt.rcParams["font.family"] = "Times New Roman"
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# %%
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
# 2215 i80_2 5088.0 1543 1548.0 1538.0 1539.0 74
df_epis = feature_set.loc[
                    (feature_set['scenario'] == 'i101_1') &
                    (feature_set['id'] == 525) &
                    (feature_set['frm'] >= 1460 - 10) &
                    (feature_set['frm'] <= 1560)]

# %%

plt.figure(figsize=(10, 7))
pos_xy = get_trajs(df_epis, df_epis.shape[0], -0.6)
plt.plot([-1, 11], [0, 0], color='black', linestyle='--')
plt.plot([-1, 11], [df_epis['pc'].max(), df_epis['pc'].max()], color='red', linestyle='--')
plt.plot(np.arange(df_epis.shape[0])*0.1, pos_xy[:, 1], linewidth = line_width, color='blue')


plt.ylim(-1.6, 4.5)
plt.xlim(0, 11 )
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.savefig("lc_extraction_example.png", dpi=500, bbox_inches='tight')
# %%
"""
Visualising data distribution
I needed this to show that our dataset is sufficiently diverse.
"""
headway_arr = feature_set['ff_long'].values[0:10000]
speed_arr = feature_set['vel'].values[0:10000]

# %%
def set_up_ax(ax):
    ax.zaxis.axes._draw_grid = False
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
        j, k = divmod(i, res)
        pl = ax.bar3d(x, y, 0, dx, dy, dz, color=color)
        pl._sort_zpos = o

fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2)
ax_headway = fig.add_subplot(1, 2, 1, projection='3d')
ax_headway.set_xlabel('Headway distance (m)', rotation=0, labelpad=20)
# ax_headway.set_zlim([0, 4000])
set_up_ax(ax_headway)

ax_speed = fig.add_subplot(1, 2, 2, projection='3d')
ax_speed.set_xlabel('Long. speed $\mathrm{(ms^{-1}}$)', rotation=0, labelpad=20)
# ax_headway.set_zlim([0, 4000])
set_up_ax(ax_speed)
################################################ Headway

################  US 101 in Los Angeles ################
################  I-80 interstate in the San Francisco Bay ################
################################################ Speed
bins = 10
plot_bar(speed_arr, ax_speed, bins, orientation='rear')
plot_bar(speed_arr, ax_speed, bins, orientation='front')
plot_bar(headway_arr, ax_headway, bins, orientation='rear')
plot_bar(headway_arr, ax_headway, bins, orientation='front')
################################################ Speed
################  US 101 in Los Angeles ################
################  I-80 interstate in the San Francisco Bay ################
# ax_headway.legend(['fa', 'scsc'])


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


ax_headway.legend(custom_lines, ['US 101 in Los Angeles', 'I-80 interstate in the San Francisco Bay Area'],
                  loc='lower center', bbox_to_anchor=(1, -0.3),
                fancybox=False, shadow=False, edgecolor=None, ncol=2, frameon=False)

################  I-80 interstate in the San Francisco, California #
plt.savefig("ngsim_state_distribution.png", dpi=500, bbox_inches='tight')
# %%
