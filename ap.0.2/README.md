# Make sure you are in the root folder of the project.
# Requirement: collect playthroughs first (see ./playthroughs/README.md).
# wget https://bit.ly/2v6nbC1

PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/train.json --out ./ap.0.2/train.seen.json --graph-type seen
PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/train.json --out ./ap.0.2/train.full.json --graph-type full

PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/valid.json --out ./ap.0.2/valid.seen.json --graph-type seen
PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/valid.json --out ./ap.0.2/valid.full.json --graph-type full

PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/test.json --out ./ap.0.2/test.seen.json --graph-type seen
PYTHONPATH=. python ./ap.0.2/make_action_prediction_dataset.py ./playthroughs/test.json --out ./ap.0.2/test.full.json --graph-type full
