# Building the datasets needed for pre-training the graph encoder by Deep Graph Infomax

GATA's graph encoder can be pre-trained using Deep Graph Infomax. We make use of trajectories extracted from playing several text-based games.

For convenience, the datasets were made available at [https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/dgi.0.2.zip](https://github.com/xingdi-eric-yuan/GATA-public/releases/download/data/dgi.0.2.zip). Once downloaded, extract its contents into `dgi.0.2/`.

## Prerequisite

Collect playthroughs (see [../playthroughs/README.md](../playthroughs/README.md)).

## Instructions

Make sure you are in the root folder of the project. Then run,

    PYTHONPATH=. python ./dgi.0.2/make_dgi_dataset.py ./playthroughs/train.json --out ./dgi.0.2/train.seen.json --graph-type seen
    PYTHONPATH=. python ./dgi.0.2/make_dgi_dataset.py ./playthroughs/train.json --out ./dgi.0.2/train.full.json --graph-type full

    PYTHONPATH=. python ./dgi.0.2/make_dgi_dataset.py ./playthroughs/valid.json --out ./dgi.0.2/valid.seen.json --graph-type seen
    PYTHONPATH=. python ./dgi.0.2/make_dgi_dataset.py ./playthroughs/valid.json --out ./dgi.0.2/valid.full.json --graph-type full

    PYTHONPATH=. python ./dgi.0.2/make_dgi_dataset.py ./playthroughs/test.json --out ./dgi.0.2/test.seen.json --graph-type seen
    PYTHONPATH=. python ./dgi.0.2/make_dgi_dataset.py ./playthroughs/test.json --out ./dgi.0.2/test.full.json --graph-type full
