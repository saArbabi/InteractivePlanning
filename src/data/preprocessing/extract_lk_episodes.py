"""
A lane keep episode in one in which:

the ego does not perform a
lane change
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from src.data.preprocessing import utils
from math import hypot
import json
from importlib import reload

cwd = os.getcwd()

# %%
"""
Note: here mveh_df is not actually merging. mveh_df is the ego car.
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

datasets = {
        "i101_1": "trajdata_i101_trajectories-0750am-0805am.txt",
        "i101_2": "trajdata_i101_trajectories-0805am-0820am.txt",
        "i101_3": "trajdata_i101_trajectories-0820am-0835am.txt",
        "i80_1": "trajdata_i80_trajectories-0400-0415.txt",
        "i80_2": "trajdata_i80_trajectories-0500-0515.txt",
        "i80_3": "trajdata_i80_trajectories-0515-0530.txt"}

col_df_all = ['id','frm','scenario','lane_id','length','x_front','y_front','class']

# %%
feature_set = pd.read_csv('./src/datasets/feature_set.txt', delimiter=' ',
                        header=None, names=col)

df_all = pd.read_csv('./src/datasets/df_all.txt', delimiter=' ',
                                                            header=None, names=col_df_all)

os.chdir('../NGSIM_data_and_visualisations')
import road_geometry
reload(road_geometry)

xc_80, yc_80 = road_geometry.get_centerlines('./NGSIM DATA/centerlines80.txt')
xc_101, yc_101 = road_geometry.get_centerlines('./NGSIM DATA/centerlines101.txt')

os.chdir(cwd)
# %%
def draw_traj(m_df, y_df, case_info):
    # for some vis
    fig = plt.figure()
    item = 'pc'
    plt.plot(m_df['frm'], m_df[item])
    # plt.plot(y_df[item])

    # plt.plot(y_df[item])
    indx = m_df.loc[m_df['frm'] == case_info['lc_frm']].index[0]

    plt.scatter(case_info['lc_frm'], m_df[item].iloc[indx])
    plt.title([case_info['id'], case_info['lc_frm'], case_info['scenario']])
    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])


    fig = plt.figure()
    item = 'act_lat'
    plt.plot(m_df['frm'], m_df[item])

    # plt.plot(y_df[item])
    # plt.plot(y_df[item])
    plt.scatter(case_info['lc_frm'], m_df[item].iloc[indx])

    plt.grid()
    plt.legend(['merge vehicle','yield vehicle'])

def get_glob_df(case_info):
    """
    :return: global pose of interacting cars
    Note: start_frm and end_frm are not included here. They are dropped later when
    calculating acceleations.
    """

    glob_pos = df_all.loc[(df_all['scenario'] == case_info['scenario']) &
                            (df_all['frm'] >= case_info['start_frm']) &
                            (df_all['frm'] <=  case_info['end_frm'])]

    return glob_pos[['id','frm','x_front','y_front', 'length']]

def get_lane_cor(scenario, lane_id):
    if scenario in ['i101_1', 'i101_2', 'i101_3']:
        xc = np.array(xc_101[int(lane_id-1)])
        yc = np.array(yc_101[int(lane_id-1)])
    else:
        xc = np.array(xc_80[int(lane_id-1)])
        yc = np.array(yc_80[int(lane_id-1)])

    return [xc, yc]

# %%
reload(utils)

def get_frm_breaks(df):
    """
    returns the index of instants at which two adjacent frms are more than 1 apart
    """
    frm_diff_index = df[df['frm'].diff() > 1].index.tolist()
    if frm_diff_index:
        return [0] + frm_diff_index + [df.shape[0]]
    else:
        return [0, df.shape[0]]

def perform_extraction():
    episode_spec = {}
    counter = 2926
    max_episdoe_count = counter * 2

    for scenario in datasets:
        feat_df = feature_set.loc[(feature_set['scenario'] == scenario) &
                                            (feature_set['lane_id'] < 7)] # feat_set_scene
        ids = feat_df['id'].unique().astype('int')

        for id in ids:
            mveh_df = feat_df.loc[(feat_df['id'] == id)].reset_index(drop = True)
            if mveh_df['e_class'].iloc[0] != 2:
                continue

            lane_ids = mveh_df['lane_id'].unique().astype('int')
            for lane_id in lane_ids:
                mveh_df_lane = mveh_df.loc[(mveh_df['lane_id'] == lane_id)].reset_index(drop = True)
                ff_ids = mveh_df_lane[mveh_df_lane['ff_id'] > 0]['ff_id'].unique().astype('int')
                for ff_id in ff_ids:
                    mveh_df_lane_ff = mveh_df_lane.loc[\
                            (mveh_df_lane['ff_id'] == ff_id)].reset_index(drop = True)

                    mveh_df_lane_ff = mveh_df_lane_ff[mveh_df_lane_ff['act_lat'].abs() < 0.1].reset_index(drop = True)
                    frm_diff_index = get_frm_breaks(mveh_df_lane_ff)
                    for i, start_i in enumerate(frm_diff_index[:-1]):
                        end_i = frm_diff_index[i+1]
                        mveh_df_epis = mveh_df_lane_ff.iloc[start_i:end_i]
                        v_min = mveh_df_epis['vel'].min()
                        frms_n = end_i-start_i
                        lc_type = 0 # indicates lane keeping
                        start_frm = mveh_df_epis['frm'].min()
                        end_frm = mveh_df_epis['frm'].max()

                        if lane_id == 1:
                            y_ids = mveh_df_epis['br_id']
                            fadj_ids = mveh_df_epis['fr_id']
                        else:
                            y_ids = mveh_df_epis['bl_id']
                            fadj_ids = mveh_df_epis['fl_id']

                        if frms_n < 30 or v_min < 0 or \
                                        y_ids.iloc[0] != y_ids.iloc[-1] or \
                                        fadj_ids.iloc[0] != fadj_ids.iloc[-1]:
                            continue

                        case_info = {
                                    'scenario':scenario,
                                    'id':id,
                                    'frms_n':frms_n,
                                    'start_frm':start_frm,
                                    'end_frm':end_frm,
                                    'episode_id': counter,
                                    'lc_type': lc_type
                                }

                        lane_cor = get_lane_cor(scenario, lane_id)
                        m_df = mveh_df_epis
                        m_df = utils.get_m_features(mveh_df_epis, case_info)
                        o_df = utils.frmTrim(feat_df, end_frm, start_frm) # other vehicles' df
                        glob_pos = get_glob_df(case_info)

                        y_id = y_ids.iloc[0]
                        fadj_id = fadj_ids.iloc[0]

                        if y_id:
                            y_df = utils.get_o_df(o_df, y_id, case_info['episode_id'])
                            m_df, y_df = utils.get_dxdv(glob_pos, m_df, y_df, lane_cor, 'front')
                            y_df['exists'] = 1

                        if fadj_id:
                            fadj_df = utils.get_o_df(o_df, fadj_id, case_info['episode_id'])
                            m_df, fadj_df = utils.get_dxdv(glob_pos, m_df, fadj_df, lane_cor, 'behind')
                            fadj_df['exists'] = 1

                        f_df = utils.get_o_df(o_df, ff_id, case_info['episode_id'])
                        m_df, f_df = utils.get_dxdv(glob_pos, m_df, f_df, lane_cor, 'behind')
                        f_df['exists'] = 1
                        f_df = utils.remove_redundants(f_df)

                        mveh_size = len(m_df)

                        if not y_id:
                            y_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                        else:
                            y_df = utils.applyCorrections(m_df, y_df, mveh_size)

                        if not fadj_id:
                            fadj_df = utils.get_dummyVals(case_info['episode_id'], mveh_size)
                        else:
                            fadj_df = utils.applyCorrections(m_df, fadj_df, mveh_size)

                        assert f_df.size == m_df.size, 'Err Mismatching df size'
                        # plt.figure()
                        # plt.plot(y_df['act_long'])
                        # plt.plot(fadj_df['act_long'])
                        # plt.grid()
                        # plt.legend(['y_df', 'fadj_df'])
                        # title = str(counter) + ' veh id: ' + str(id) + ' scen: ' + scenario
                        # plt.title(title)
                        if counter == max_episdoe_count:
                            print('Extraction compete')
                            return

                        else:
                            utils.data_saver(m_df, 'm_df_lk')
                            utils.data_saver(y_df, 'y_df_lk')
                            utils.data_saver(f_df, 'f_df_lk')
                            utils.data_saver(fadj_df, 'fadj_df_lk')
                            print(counter, ' ### lane change extracted ###')

                        counter += 1
perform_extraction()
# %%
2 * 2
165 * 15 / 60
