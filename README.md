# My Paper Title

This repository is the official implementation of InteractivePlanning
<!-- # [I](https://arxiv.org/abs/2030.12345). -->

## Training

To train the model(s) in the paper, go to the models/model_trainers folder and
run the the script for the any of the models presented in the paper.

## Evaluation

To evaluate the models using the Root Weighted Square Error (RWSE) metric, run
the eval_run.py script. The script will fetch the dataset, load the relevant models and
then runs monte carlo simulations. The simulation logs (vehicle states) are saved
in the model folder.

## Data Preprocessing
Here is a brief description of the path from raw NGSIM data to model.

- feature_set.txt contains the extracted features from the filtered NGSIM dataset
- df_all.txt is raw data for the entire dataset.
- lc episodes are extracted for x4 cars. These are m_df, y_df, f_df and fadj_df.
- These data files are then trimmed and saved as states_arr and targets_arr by
data_preliminaries.py to be further processed/sequenced prior to training by
data_prep.py
- the relevant data files are then saved into a folder with a unique id and config file.
The train and validation sets are already scaled.
 
