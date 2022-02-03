# My Paper Title

This repository is the official implementation of InteractivePlanning
<!-- # [I](https://arxiv.org/abs/2030.12345). -->

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Training

To train the model(s) in the paper, go to the models/model_trainers folder and
run the the script for the any of the models presented in the paper.

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate the models using the Root Weighted Square Error (RWSE), run
the eval_run.py script. The script will fetch the dataset, load the relevant models and
then runs monte carlo simulations. The simulation logs (vehicle states) are saved
in the model folder.
 
