# Building the datasets needed for pre-training the graph updater on a Command Generation task

GATA's graph updater can be pre-trained by a Command Generation task. We make use of the set of cooking games available at [https://aka.ms/ftwp/dataset.zip](https://aka.ms/ftwp/dataset.zip) that were used for the [First TextWorld Problem competition](https://aka.ms/ftwp).

For convenience, the datasets were made available at [https://aka.ms/twkg/cmd_gen.0.2.zip](https://aka.ms/twkg/cmd_gen.0.2.zip). Once downloaded, extract its contents into `cmd_gen.0.2/`.

## Instructions

Make sure you are in the root folder of the project. Then run,

    PYTHONPATH=. python ./cmd_gen.0.2/make_cmd_gen_dataset.py --branching 10 --games-dir ./games/train/ --out ./cmd_gen.0.2/train.playthroughs.json && PYTHONPATH=. python ./cmd_gen.0.2/compress_cmd_gen_dataset.py ./cmd_gen.0.2/train.playthroughs.json --out ./cmd_gen.0.2/train.json
    PYTHONPATH=. python ./cmd_gen.0.2/make_cmd_gen_dataset.py --branching 10 --games-dir ./games/valid/ --out ./cmd_gen.0.2/valid.playthroughs.json && PYTHONPATH=. python ./cmd_gen.0.2/compress_cmd_gen_dataset.py ./cmd_gen.0.2/valid.playthroughs.json --out ./cmd_gen.0.2/valid.json
    PYTHONPATH=. python ./cmd_gen.0.2/make_cmd_gen_dataset.py --branching 10 --games-dir ./games/test/ --out ./cmd_gen.0.2/test.playthroughs.json && PYTHONPATH=. python ./cmd_gen.0.2/compress_cmd_gen_dataset.py ./cmd_gen.0.2/test.playthroughs.json --out ./cmd_gen.0.2/test.json

> :exclamation: The script will attempt to download (~1.8 GB) and extract the games in `./games/`.
