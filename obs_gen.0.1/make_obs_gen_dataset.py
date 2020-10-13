import os
import json
import argparse
from os.path import join as pjoin

import tqdm
import numpy as np

import textworld

from graph_dataset import GraphDataset


def next_example(datapoints):
    # This function assume `datapoints` is sequential wrt point['step'].
    # e.g. (0, 0) ... (0, M) ... (N, 0) ...(N, M)
    walkthrough = []

    data = []
    for point in tqdm.tqdm(datapoints):
        if (walkthrough and walkthrough[-1]["game"] != point["game"]):
            walkthrough = []  # New game.

        if point["step"][1] == 0:
            walkthrough.append(point)
            data = []
        else:
            data.append(point)

        yield walkthrough + data


def build_observation_generation_dataset(args):
    # Expected key for each datapoint (from cmd_gen.0.2):
    # game, step, observation, previous_action, target_commands, previous_graph_seen
    with open(args.input) as f:
        datapoints = json.load(f)["examples"]

    dataset = []
    for sequence in next_example(datapoints):
        # For each data point we want the following sequence of 6 keys:
        # game, step, observation, previous_action
        dataset.append([
            {
                "game": example["game"],
                "step": example["step"],
                "observation": example["observation"],
                "previous_action": example["previous_action"],
            }
            for example in sequence
        ])

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "obs_gen.json"

    with open(args.output, "w") as f:
        json.dump(dataset, f)

    if args.verbose:
        print("This dataset has {:,} datapoints.".format(len(dataset)))


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", default="dataset.json",
                        help="Path where to load the dataset (.json)")

    parser.add_argument("--output",
                        help="Path where to save the dataset (.json)")

    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing files.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.output and os.path.isfile(args.output) and not args.force:
        parser.error("{} already exists. Use -f to overwrite.".format(args.output))

    build_observation_generation_dataset(args)

if __name__ == "__main__":
    main()
