from src.data.preprocessing import data_prep
from src.data.preprocessing import data_obj
reload(data_prep)
reload(data_obj)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from importlib import reload


DataPrep = data_prep.DataPrep
DataObj = data_obj.DataObj

config = {
 "model_config": {
     "learning_rate": 1e-3,
     "neurons_n": 50,
     "layers_n": 2,
     "epochs_n": 50,
     "components_n": 5
},
"data_config": {"obs_n": 20,
                "pred_step_n": 10,
                "step_size": 1,
                "Note": ""
                # "Note": "jerk as target"
},
"exp_id": "NA",
"Note": "NA"
}
data_objs =  DataObj(config).loadData()
states_train, targets_train, conditions_train, \
                            states_val, targets_val, conditions_val = data_objs

states_, targets_, conditions_ = data_objs[0:3] # train

# %%
sum([2, 3])
# %%
conditions_train[10][0].shape
plt.plot(states_train[10][15, :, 3], color='black')
plt.plot(range(19, 29), targets_train[10][0][15, :, 1], color='red')
plt.plot(range(19, 29), conditions_train[10][0][15, :, 1], color='red')

# %%
size = 0
for i in targets_train.keys():
    size += states_train[i].shape[0]
size

# %%
states_train[4].shape
# %%
plt.plot(states_train[17][0, :, 3])
plt.plot(states_train[17][0, :, 3]+np.random.normal(0, 0.01, 20))
# %%
for i in range(0, 5):
    plt.figure()
    plt.hist(conditions_val[4][i][:,0,:], bins=125)
# %%
for i in range(0, 5):
    plt.figure()
    plt.hist(targets_train[20][i][:,0,:], bins=125)

# %%
for i in range(0, 17):
    plt.figure()
    plt.hist(states_train[4][0:10000,1,i], bins=125)
# %%
