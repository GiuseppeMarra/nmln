# Neural Markov Logic Networks

This branch contains the code necessary to replicate the experiments in the paper.
This brunch will differ from the master brunch where we will keep the updated version of the library.



## Installation

Just run:

    pip intall -e ./


## Run experiments

The experiments are in two folders:

- `kbc`, where we included the experiments for the Knowledge Base completion tasks
- `molecules`, where we included the experiments for the Molecule Generation task.

To run an experiment, just run the corresponding script from the folder, e.g.

    cd molecules
    python generation_with_constraints.py



Hyperparameters can be set directly in the corresponding scripts. 
