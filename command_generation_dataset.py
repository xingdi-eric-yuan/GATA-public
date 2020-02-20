import os
import json
from tqdm import tqdm
from os.path import join as pjoin

import numpy as np
import gym

from generic import sort_target_commands
# from generic import process_equivalent_entities_in_triplet, process_equivalent_entities_in_command
# from generic import process_burning_triplets, process_burning_commands, process_direction_triplets, process_direction_commands, arguments_swap
# from generic import process_exits_in_triplet
# from generic import two_args_relations, one_arg_state_relations, ignore_relations
from graph_dataset import GraphDataset


class CommandGenerationData(gym.Env):

    FILENAMES_MAP = {
        "train": "train.json",
        "valid": "valid.json",
        "test": "test.json"
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
                "observation_strings": [],
                "previous_triplets": [],
                "target_commands": [],
                }
            self.load_dataset_for_cmd_gen(split)

        print("loaded dataset from {} ...".format(self.data_path))
        self.train_size = len(self.dataset["train"]["observation_strings"])
        self.valid_size = len(self.dataset["valid"]["observation_strings"])
        self.test_size = len(self.dataset["test"]["observation_strings"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def load_dataset_for_cmd_gen(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        desc = "Loading {}".format(os.path.basename(file_path))
        print(desc)
        with open(file_path) as f:
            data = json.load(f)

        graph_dataset = GraphDataset.loads(data["graph_index"])
        self.dataset[split]["graph_dataset"] = graph_dataset

        for example in tqdm(data["examples"], desc=desc):
            observation = "{feedback} <sep> {action}".format(feedback=example["observation"],
                                                             action=example["previous_action"])
            # Need to sort target commands to enable the seq2seq model to learn the ordering.
            target_commands = " <sep> ".join(sort_target_commands(example["target_commands"]))

            self.dataset[split]["observation_strings"].append(observation)
            self.dataset[split]["previous_triplets"].append(example["previous_graph_seen"])
            self.dataset[split]["target_commands"].append(target_commands)

    def read_config(self):
        self.data_path = self.config["cmd_gen"]["data_path"]
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
            self.data = {"observation_strings": self.dataset[split]["observation_strings"][: self.use_this_many_data],
                         "previous_triplets": self.dataset[split]["previous_triplets"][: self.use_this_many_data],
                         "target_commands": self.dataset[split]["target_commands"][: self.use_this_many_data]}
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

        observation_strings, previous_triplets, target_commands = [], [], []
        decompress = self.dataset[self.split]["graph_dataset"].decompress
        for idx in indices:
            observation_strings.append(self.data["observation_strings"][idx])
            previous_triplets.append(decompress(self.data["previous_triplets"][idx]))
            target_commands.append(self.data["target_commands"][idx])

        return observation_strings, previous_triplets, target_commands

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

