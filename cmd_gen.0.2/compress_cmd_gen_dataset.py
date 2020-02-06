import os
import json
import argparse
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np

import textworld

from graph_dataset import GraphDataset


def compress_command_generation_dataset(args):
    # Expected key for each playthrough:
    # game, step, observation, previous_action, target_commands, previous_graph_seen, graph_seen
    playthroughs = (json.loads(line.rstrip(",\n")) for line in open(args.input) if len(line.strip()) > 1)

    graph_dataset = GraphDataset()
    dataset = []
    for example in tqdm(playthroughs):
        previous_graph_seen = graph_dataset.compress(example["previous_graph_seen"])
        target_commands = example["target_commands"]

        # For each data point we want the following 6 keys:
        # game, step, observation, previous_action, target_commands, previous_graph_seen
        dataset.append({
            "game": example["game"],
            "step": example["step"],
            "observation": example["observation"],
            "previous_action": example["previous_action"],
            "previous_graph_seen": previous_graph_seen,
            "target_commands": example["target_commands"],
        })

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".cmd_gen.json"

    data = {
        "graph_index": graph_dataset.dumps(),
        "examples": dataset,
    }
    with open(args.output, "w") as f:
        json.dump(data, f)

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

    if os.path.isfile(args.output) and not args.force:
        parser.error("{} already exists. Use -f to overwrite.".format(args.output))

    compress_command_generation_dataset(args)

if __name__ == "__main__":
    main()
