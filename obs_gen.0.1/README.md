# Make sure you are in the root folder of the project.
# Requirement: Download the cmd_gen_0.2 dataset (see cmd_gen.0.2/README.md)

# wget https://bit.ly/3ep1yhI

PYTHONPATH=. python ./obs_gen.0.1/make_obs_gen_dataset.py ./cmd_gen.0.2/train.json --out ./obs_gen.0.1/train.json
PYTHONPATH=. python ./obs_gen.0.1/make_obs_gen_dataset.py ./cmd_gen.0.2/valid.json --out ./obs_gen.0.1/valid.json
PYTHONPATH=. python ./obs_gen.0.1/make_obs_gen_dataset.py ./cmd_gen.0.2/test.json --out ./obs_gen.0.1/test.json
