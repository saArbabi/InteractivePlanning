import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# %%
spec_col = ['episode_id', 'scenario', 'lc_frm', 'm_id', 'y_id', 'fadj_id', 'f_id',
       'frm_n']
state_col = ['episode_id', 'vel', 'pc', 'act_long','act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'vel', 'dx', 'act_long', 'act_lat',
                                     'lc_type', 'exists', 'exists', 'exists']

validation_episodes = np.loadtxt('./src/datasets/validation_episodes.csv', delimiter=',')
all_state_arr = np.loadtxt('./src/datasets/states_arr.csv', delimiter=',')
validation_set = np.loadtxt('./src/datasets/states_val_arr.csv', delimiter=',')
all_target_arr = np.loadtxt('./src/datasets/targets_arr.csv', delimiter=',')
spec = pd.read_csv('./src/datasets/episode_spec.txt', delimiter=' ',
                                                        header=None, names=spec_col)

# %%
def pickup_episodes(validation_set, min_speed, max_speed, episode_n):
    # traffic density is roughly assumed to be indicated by the average vehicle speeds
    potential_episodes = validation_set[np.where(
                                        (validation_set[:, 1]<max_speed) &
                                        (validation_set[:, 1]>min_speed) &
                                        (validation_set[:, 5]<max_speed) &
                                        (validation_set[:, 5]>min_speed) &
                                        (validation_set[:, 9]<max_speed) &
                                        (validation_set[:, 9]>min_speed) &
                                        (validation_set[:, 13]<max_speed) &
                                        (validation_set[:, 13]>min_speed))][:, 0]

    possible_episodes  = spec.loc[(spec['frm_n']>40) &
                                    (spec['y_id']!=0) &
                                    (spec['f_id']!=0) &
                                    (spec['fadj_id']!=0)]['episode_id'].values


    episodes = possible_episodes[np.isin(possible_episodes, np.unique(potential_episodes))]
    print(episodes.shape)

    return np.random.choice(episodes, episode_n, replace=False)

def data_saver(data, data_name):
    file_name = './src/datasets/' + data_name + '.csv'
    if data.dtype == 'int64':
        np.savetxt(file_name, data, fmt='%i', delimiter=',')
    else:
        np.savetxt(file_name, data, fmt='%10.3f', delimiter=',')

# %%
# low_density_episodes = pickup_episodes(validation_set, min_speed=12, max_speed=25, episode_n=50)
medium_density_episodes = pickup_episodes(validation_set, min_speed=7, max_speed=14, episode_n=50)
high_density_episodes = pickup_episodes(validation_set, min_speed=0, max_speed=7, episode_n=47)

#
#
_arr = all_state_arr[np.isin(all_state_arr[:, 0], medium_density_episodes)]
plt.hist(_arr[:, 1])
_arr = all_state_arr[np.isin(all_state_arr[:, 0], low_density_episodes)]
plt.hist(_arr[:, 1])
_arr = all_state_arr[np.isin(all_state_arr[:, 0], high_density_episodes)]
plt.hist(_arr[:, 1])
# %%
data_saver(medium_density_episodes, 'medium_density_val_episodes')
# data_saver(low_density_episodes, 'low_density_test_episodes')
data_saver(high_density_episodes, 'high_density_val_episodes')
