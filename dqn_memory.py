import random
from collections import namedtuple
import numpy as np
import torch


# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('observation_list', 'action_candidate_list', 'chosen_indices', 'graph_triplets', 'reward', 'graph_reward', 'is_final'))

class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, discount_gamma_game_reward=1.0, discount_gamma_graph_reward=1.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.discount_gamma_game_reward = discount_gamma_game_reward
        self.discount_gamma_graph_reward = discount_gamma_graph_reward
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_prior=False, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def get_next_final_pos(self, which_memory, head):
        i = head
        while True:
            if i >= len(which_memory):
                return None
            if which_memory[i].is_final:
                return i
            i += 1
        return None

    def _get_single_transition(self, n, which_memory):
        assert n > 0
        tried_times = 0
        while True:
            tried_times += 1
            if tried_times >= 50:
                return None
            if len(which_memory) <= n:
                return None
            head = np.random.randint(0, len(which_memory) - n)
            # if n is 1, then head can't be is_final
            if n == 1:
                if which_memory[head].is_final:
                    continue
            #  if n > 1, then all except tail can't be is_final
            else:
                if np.any([item.is_final for item in which_memory[head: head + n]]):
                    continue
                    
            next_final = self.get_next_final_pos(which_memory, head)
            if next_final is None:
                continue

            # all good
            obs = which_memory[head].observation_list
            candidate = which_memory[head].action_candidate_list
            chosen_indices = which_memory[head].chosen_indices
            graph_triplets = which_memory[head].graph_triplets

            next_obs = which_memory[head + n].observation_list
            next_candidate = which_memory[head + n].action_candidate_list
            next_graph_triplets = which_memory[head + n].graph_triplets

            rewards_up_to_next_final = [self.discount_gamma_game_reward ** i * which_memory[head + i].reward for i in range(next_final - head + 1)]
            reward = torch.sum(torch.stack(rewards_up_to_next_final))

            graph_rewards_up_to_next_final = [self.discount_gamma_graph_reward ** i * which_memory[head + i].graph_reward for i in range(next_final - head + 1)]
            graph_reward = torch.sum(torch.stack(graph_rewards_up_to_next_final))

            return (obs, candidate, chosen_indices, graph_triplets, reward + graph_reward, next_obs, next_candidate, next_graph_triplets, n)

    def _get_batch(self, n_list, which_memory):
        res = []
        for i in range(len(n_list)):
            output = self._get_single_transition(n_list[i], which_memory)
            if output is None:
                continue
            res.append(output)

        if len(res) == 0:
            return None
        return res

    def get_batch(self, batch_size, multi_step=1):
        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))
        res = []
        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch(np.random.randint(1, multi_step + 1, size=from_alpha), self.alpha_memory)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch(np.random.randint(1, multi_step + 1, size=from_beta), self.beta_memory)
        if res_alpha is None and res_beta is None:
            return None
        if res_alpha is not None:
            res += res_alpha
        if res_beta is not None:
            res += res_beta
        random.shuffle(res)

        obs_list, candidate_list, chosen_indices_list, graph_triplet_list, reward_list, actual_n_list = [], [], [], [], [], []
        next_obs_list, next_candidate_list, next_graph_triplet_list = [], [], []

        for item in res:
            
            obs, candidate, chosen_indices, graph_triplets, reward, next_obs, next_candidate, next_graph_triplets, n = item
            obs_list.append(obs)
            candidate_list.append(candidate)
            chosen_indices_list.append(chosen_indices)
            graph_triplet_list.append(graph_triplets)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            next_candidate_list.append(next_candidate)
            next_graph_triplet_list.append(next_graph_triplets)
            actual_n_list.append(n)

        indices = np.array(chosen_indices_list)  # batch
        rewards = torch.stack(reward_list, 0)  # batch
        actual_n_list = np.array(actual_n_list)

        return obs_list, candidate_list, indices, graph_triplet_list, rewards, next_obs_list, next_candidate_list, next_graph_triplet_list, actual_n_list

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
