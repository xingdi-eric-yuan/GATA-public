# Make sure you are in the root folder of the project.
# If needed, the script will download and extract the FTWP competition games automatically.

# wget https://bit.ly/385UVOC

PYTHONPATH=. python ./cmd_gen.0.2/make_cmd_gen_dataset.py --branching 10 --games-dir ./games/train/ --out ./cmd_gen.0.2/train.playthroughs.json && PYTHONPATH=. python ./cmd_gen.0.2/compress_cmd_gen_dataset.py ./cmd_gen.0.2/train.playthroughs.json --out ./cmd_gen.0.2/train.json
PYTHONPATH=. python ./cmd_gen.0.2/make_cmd_gen_dataset.py --branching 10 --games-dir ./games/valid/ --out ./cmd_gen.0.2/valid.playthroughs.json && PYTHONPATH=. python ./cmd_gen.0.2/compress_cmd_gen_dataset.py ./cmd_gen.0.2/valid.playthroughs.json --out ./cmd_gen.0.2/valid.json
PYTHONPATH=. python ./cmd_gen.0.2/make_cmd_gen_dataset.py --branching 10 --games-dir ./games/test/ --out ./cmd_gen.0.2/test.playthroughs.json && PYTHONPATH=. python ./cmd_gen.0.2/compress_cmd_gen_dataset.py ./cmd_gen.0.2/test.playthroughs.json --out ./cmd_gen.0.2/test.json
