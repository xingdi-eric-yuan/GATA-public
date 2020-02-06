import os
import json
from os.path import join as pjoin

from tqdm import tqdm

import numpy as np
import gym

from graph_dataset import GraphDataset


class SPData(gym.Env):

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
                "target_graph": [],
                "previous_graph": [],
                "action": [],
                "graph_choices": []
                }
            self.load_dataset_for_sp(split)

        print("loaded dataset from {} ...".format(self.data_path))
        self.train_size = len(self.dataset["train"]["target_graph"])
        self.valid_size = len(self.dataset["valid"]["target_graph"])
        self.test_size = len(self.dataset["test"]["target_graph"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def load_dataset_for_sp(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[self.graph_type][split])
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        self.dataset[split]["graph_dataset"] = graph_dataset

        desc = "Loading {}".format(os.path.basename(file_path))
        for example in tqdm(data["examples"], desc=desc):
            action = example["action"]
            target_graph = example["target_graph"]
            prev_graph = example["previous_graph"]
            graph_choices = example["graph_choices"]

            self.dataset[split]["target_graph"].append(target_graph)
            self.dataset[split]["previous_graph"].append(prev_graph)
            self.dataset[split]["action"].append(action)
            self.dataset[split]["graph_choices"].append(graph_choices)

    def read_config(self):
        self.data_path = self.config["sp"]["data_path"]
        self.graph_type = self.config["sp"]["graph_type"]
        if self.config["general"]["philly"]:
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
            self.data = {"target_graph": self.dataset[split]["target_graph"][: self.use_this_many_data],
                         "previous_graph": self.dataset[split]["previous_graph"][: self.use_this_many_data],
                         "action": self.dataset[split]["action"][: self.use_this_many_data],
                         "graph_choices": self.dataset[split]["graph_choices"][: self.use_this_many_data]}
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

        target_graph, previous_graph, action, graph_choices = [], [], [], []
        decompress = self.dataset[self.split]["graph_dataset"].decompress
        for idx in indices:
            action.append(self.data["action"][idx])
            # Perform just-in-time decompression.
            target_graph.append(decompress(self.data["target_graph"][idx]))
            previous_graph.append(decompress(self.data["previous_graph"][idx]))
            graph_choices.append([decompress(idx_) for idx_ in self.data["graph_choices"][idx]])

        return target_graph, previous_graph, action, graph_choices

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
