# Building the datasets needed for pre-training the graph updater on an Observation Generation or a Contrastive Observation Classification task

GATA's graph updater can be pre-trained by an Observation Generation or a Contrastive Observation Classification task. We make use of trajectories extracted from playing several text-based games.

For convenience, the datasets were made available at [https://aka.ms/twkg/obs_gen.0.1.zip](https://aka.ms/twkg/obs_gen.0.1.zip). Once downloaded, extract its contents into `obs_gen.0.1/`.

## Prerequisite

Generate the cmd_gen.0.2 datasets (see [../cmd_gen.0.2/README.md](../cmd_gen.0.2/README.md)).

## Instructions

Make sure you are in the root folder of the project. Then run,

    PYTHONPATH=. python ./obs_gen.0.1/make_obs_gen_dataset.py ./cmd_gen.0.2/train.json --out ./obs_gen.0.1/train.json
    PYTHONPATH=. python ./obs_gen.0.1/make_obs_gen_dataset.py ./cmd_gen.0.2/valid.json --out ./obs_gen.0.1/valid.json
    PYTHONPATH=. python ./obs_gen.0.1/make_obs_gen_dataset.py ./cmd_gen.0.2/test.json --out ./obs_gen.0.1/test.json
