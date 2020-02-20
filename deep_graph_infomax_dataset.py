import os
import json
from os.path import join as pjoin

from tqdm import tqdm

import numpy as np
import gym

from graph_dataset import GraphDataset


class DGIData(gym.Env):

    FILENAMES_MAP = {
        "full": {
            "train": "train.full.json",
            "valid": "valid.full.json",
            "test": "test.full.json"
            },
        "seen": {
            "train": "train.seen.json",
            "valid": "valid.seen.json",
            "test": "test.seen.json"
            }
        }

    def __init__(self, config):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)

        # Load dataset splits.
        self.dataset = {}
        for split in ["train", "valid", "test"]:
            self.dataset[split] = {
                "graph": [],
                }
            self.load_dataset_for_dgi(split)

        print("loaded dataset from {} ...".format(self.data_path))
        self.train_size = len(self.dataset["train"]["graph"])
        self.valid_size = len(self.dataset["valid"]["graph"])
        self.test_size = len(self.dataset["test"]["graph"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def load_dataset_for_dgi(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[self.graph_type][split])
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        self.dataset[split]["graph_dataset"] = graph_dataset

        desc = "Loading {}".format(os.path.basename(file_path))
        for example in tqdm(data["examples"], desc=desc):
            graph = example["graph"]
            self.dataset[split]["graph"].append(graph)

    def read_config(self):
        self.data_path = self.config["dgi"]["data_path"]
        self.graph_type = self.config["dgi"]["graph_type"]

        self.data_path = os.environ['PT_DATA_DIR'] + "/" + self.data_path

        self.random_seed = self.config["general"]["random_seed"]
        self.use_this_many_data = self.config["general"]["use_this_many_data"]

        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

    def split_reset(self, split):
        if split == "train":
            self.data_size = self.train_size
            self.batch_size = self.training_batch_size
        elif split == "valid":
            self.data_size = self.valid_size
            self.batch_size = self.evaluate_batch_size
        else:
            self.data_size = self.test_size
            self.batch_size = self.evaluate_batch_size

        if split == "train" and self.use_this_many_data > 0:
            self.data = {"graph": self.dataset[split]["graph"][: self.use_this_many_data]}
            self.data_size = self.use_this_many_data
        else:
            self.data = self.dataset[split]

        self.split = split
        self.batch_pointer = 0

    def get_batch(self):
        if self.split == "train":
            indices = self.rng.choice(self.data_size, self.training_batch_size)
        else:
            start = self.batch_pointer
            end = min(start + self.training_batch_size, self.data_size)
            indices = np.arange(start, end)
            self.batch_pointer += self.training_batch_size

            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0

        graph =[]
        decompress = self.dataset[self.split]["graph_dataset"].decompress
        for idx in indices:
            graph.append(decompress(self.data["graph"][idx]))

        return graph

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
