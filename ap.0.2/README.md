# Building the datasets needed for pre-training the graph encoder on a Action Prediction task

GATA's graph encoder can be pre-trained by a Action Prediction task. We make use of trajectories extracted from playing several text-based games.

For convenience, the datasets were made available at [https://aka.ms/twkg/ap.0.2.zip](https://aka.ms/twkg/ap.0.2.zip). Once downloaded, extract its contents into `ap.0.2/`.

## Prerequisite

Collect playthroughs (see [../playthroughs/README.md](../playthroughs/README.md)).

## Instructions

Make sure you are in the root folder of the project. Then run,

    PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/train.json --out ./ap.0.2/train.seen.json --graph-type seen
    PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/train.json --out ./ap.0.2/train.full.json --graph-type full

    PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/valid.json --out ./ap.0.2/valid.seen.json --graph-type seen
    PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/valid.json --out ./ap.0.2/valid.full.json --graph-type full

    PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/test.json --out ./ap.0.2/test.seen.json --graph-type seen
    PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/test.json --out ./ap.0.2/test.full.json --graph-type full
