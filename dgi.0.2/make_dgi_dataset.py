import os
import json
import argparse
from os.path import join as pjoin

import tqdm
import numpy as np

import textworld

from graph_dataset import GraphDataset


def build_dgi_dataset(args):
    # Expected key for each playthrough:
    # game, step, action, graph_local, graph_seen, graph_full,
    playthroughs = (json.loads(line.rstrip(",\n")) for line in open(args.input) if len(line.strip()) > 1)

    graph_dataset = GraphDataset()
    dataset = []
    for example in playthroughs:
        # For each data point we want the following 3 keys:
        # game, step, graph
        dataset.append({
            "game": example["game"],
            "step": example["step"],
            "graph": graph_dataset.compress(example["graph_{}".format(args.graph_type)]),
        })

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".dgi.{}.json".format(args.graph_type)

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
    parser.add_argument("--graph-type", choices=["full", "seen"], default="full",
                        help="Specified which type of graph to use.")

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

    build_dgi_dataset(args)

if __name__ == "__main__":
    main()
