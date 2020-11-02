# Collecting playthroughs from the FTWP competition games

GATA's requires different datasets in order to pre-train its different modules. We make use of the set of cooking games available at [https://aka.ms/ftwp/dataset.zip](https://aka.ms/ftwp/dataset.zip) that were used for the [First TextWorld Problem competition](https://aka.ms/ftwp).

## Instructions

Make sure you are in the root folder of the project. Then run,

    PYTHONPATH=. python ./playthroughs/collect_playthroughs.py --games-dir ./games/train/ --out ./playthroughs/train.json
    PYTHONPATH=. python ./playthroughs/collect_playthroughs.py --games-dir ./games/valid/ --out ./playthroughs/valid.json
    PYTHONPATH=. python ./playthroughs/collect_playthroughs.py --games-dir ./games/test/ --out ./playthroughs/test.json

> :exclamation: The script will attempt to download (~1.8 GB) and extract the games in `./games/`.
