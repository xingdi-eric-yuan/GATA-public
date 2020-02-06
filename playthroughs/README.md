# Make sure you are in the root folder of the project.

# If needed, the script will download and extract the FTWP competition games automatically.

PYTHONPATH=. python ./playthroughs/collect_playthroughs.py --games-dir ./games/train/ --out ./playthroughs/train.json
PYTHONPATH=. python ./playthroughs/collect_playthroughs.py --games-dir ./games/valid/ --out ./playthroughs/valid.json
PYTHONPATH=. python ./playthroughs/collect_playthroughs.py --games-dir ./games/test/ --out ./playthroughs/test.json
