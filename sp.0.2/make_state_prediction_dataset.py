import os
import json
import argparse
from os.path import join as pjoin

import tqdm
import numpy as np

import textworld

from graph_dataset import GraphDataset


def next_example(playthroughs):
    data = []
    for point in tqdm.tqdm(playthroughs):
        if len(data) == 0:
            data.append(point)
            continue

        if (data[-1]["game"] != point["game"] or
            data[-1]["step"][0] != point["step"][0]):
                yield data
                data = []

        data.append(point)

    yield data


def build_state_prediction_dataset(args):
    # Expected key for each playthrough:
    # game, step, action, graph_local, graph_seen, graph_full,
    playthroughs = (json.loads(line.rstrip(",\n")) for line in open(args.input) if len(line.strip()) > 1)

    graph_dataset = GraphDataset()
    dataset = []
    for example in next_example(playthroughs):
        root, candidates = example[0], example[1:]
        if len(candidates) < args.min_candidates:
            continue

        previous_graph = graph_dataset.compress(root["graph_{}".format(args.graph_type)])
        graph_choices = [graph_dataset.compress(candidate["graph_{}".format(args.graph_type)])
                         for candidate in candidates]

        # For each data point we want the following 6 keys:
        # game, step, previous_graph, action, target_graph, graph_choices
        for i, candidate in enumerate(candidates):
            dataset.append({
                "game": candidate["game"],
                "step": candidate["step"],
                "action": candidate["action"],
                "previous_graph": previous_graph,
                "target_graph": graph_choices[i],
                "graph_choices": graph_choices,
            })

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".sp.{}.json".format(args.graph_type)

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
    parser.add_argument("--min-candidates", type=int, default=5,
                        help="Keep datapoint with at least that many candidates. Default: %(default)s.")

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

    build_state_prediction_dataset(args)

if __name__ == "__main__":
    main()
