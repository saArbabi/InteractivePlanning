from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

dirName = './src/datasets/'
all_state_arr = np.loadtxt(dirName+'states_arr.csv', delimiter=',')
all_target_arr = np.loadtxt(dirName+'targets_arr.csv', delimiter=',')
state_scaler = StandardScaler().fit(all_state_arr[:, 1:-4])
action_scaler = StandardScaler().fit(all_target_arr[:, 1:])

with open(dirName+'/state_scaler', "wb") as f:
    pickle.dump(state_scaler, f)

with open(dirName+'action_scaler', "wb") as f:
    pickle.dump(action_scaler, f)
