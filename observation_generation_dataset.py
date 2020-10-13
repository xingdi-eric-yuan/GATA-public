import os
import json
import random
from tqdm import tqdm
from os.path import join as pjoin

import numpy as np
import gym


class ObservationGenerationData(gym.Env):

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
                "observations": [],
                "previous_actions": [],
                }
            self.load_dataset_for_obs_gen(split)

        print("loaded dataset from {} ...".format(self.data_path))
        self.train_size = len(self.dataset["train"]["observations"])
        self.valid_size = len(self.dataset["valid"]["observations"])
        self.test_size = len(self.dataset["test"]["observations"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def load_dataset_for_obs_gen(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[split])
        desc = "Loading {}".format(os.path.basename(file_path))
        print(desc)
        with open(file_path) as f:
            data = json.load(f)

        obs_list, prev_action_list = [], []
        for sequence in tqdm(data, desc=desc):
            obs_list.append([e["observation"] for e in sequence])
            prev_action_list.append([e["previous_action"] for e in sequence])
        ids = np.arange(len(obs_list))
        random.seed(123)
        random.shuffle(ids)
        for i in range(len(ids)):
            self.dataset[split]["observations"].append(obs_list[ids[i]])
            self.dataset[split]["previous_actions"].append(prev_action_list[ids[i]])

    def read_config(self):
        self.data_path = self.config["obs_gen"]["data_path"]

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
            self.data = {"observations": self.dataset[split]["observations"][: self.use_this_many_data],
                         "previous_actions": self.dataset[split]["previous_actions"][: self.use_this_many_data]}
            self.data_size = self.use_this_many_data
        elif split == "train":
            self.data = self.dataset[split]
        else:
            # valid and test, we use 1k data points
            self.data = {"observations": self.dataset[split]["observations"][: 1000],
                         "previous_actions": self.dataset[split]["previous_actions"][: 1000]}
            self.data_size = 1000

        self.split = split
        self.batch_pointer = 0

    def get_batch(self):
        if self.split == "train":
            indices = self.rng.choice(self.data_size, self.batch_size)
        else:
            start = self.batch_pointer
            end = min(start + self.batch_size, self.data_size)
            indices = np.arange(start, end)
            self.batch_pointer += self.batch_size

            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0

        observations, previous_actions = [], []
        for idx in indices:
            observations.append(self.data["observations"][idx])
            previous_actions.append(self.data["previous_actions"][idx])

        return observations, previous_actions

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
