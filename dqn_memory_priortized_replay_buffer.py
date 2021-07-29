# part of the code are from https://github.com/hill-a/stable-baselines/
import random
from collections import namedtuple
import numpy as np
import torch
from generic import to_np
from segment_tree import SumSegmentTree, MinSegmentTree


# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('observation_list', 'prev_action_list', 'action_candidate_list', 'chosen_indices', 'graph_triplets', 'reward', 'graph_reward', 'count_reward', 'is_final'))

class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0, discount_gamma_game_reward=1.0, discount_gamma_graph_reward=1.0, discount_gamma_count_reward=1.0, accumulate_reward_from_final=False, seed=None):
        self.rng = np.random.RandomState(seed)

        # prioritized replay memory
        self._storage = []
        self.capacity = capacity
        self._next_idx = 0

        assert priority_fraction >= 0
        self._alpha = priority_fraction

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.discount_gamma_game_reward = discount_gamma_game_reward
        self.discount_gamma_graph_reward = discount_gamma_graph_reward
        self.discount_gamma_count_reward = discount_gamma_count_reward
        self.accumulate_reward_from_final = accumulate_reward_from_final

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self.capacity

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, *args):
        """
        add a new transition to the buffer
        """
        idx = self._next_idx
        data = Transition(*args)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self.capacity
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def get_next_final_pos(self, which_memory, head):
        i = head
        while True:
            if i >= len(self._storage):
                return None
            if self._storage[i].is_final:
                return i
            i += 1
        return None

    def _get_single_transition(self, idx, n):
        assert n > 0
        head = idx
        # if n is 1, then head can't be is_final
        if n == 1:
            if self._storage[head].is_final:
                return None
        #  if n > 1, then all except tail can't be is_final
        else:
            if np.any([item.is_final for item in self._storage[head: head + n]]):
                return None

        next_final = self.get_next_final_pos(self._storage, head)
        if next_final is None:
            return None

        # all good
        obs = self._storage[head].observation_list
        prev_action = self._storage[head].prev_action_list
        candidate = self._storage[head].action_candidate_list
        chosen_indices = self._storage[head].chosen_indices
        graph_triplets = self._storage[head].graph_triplets

        next_obs = self._storage[head + n].observation_list
        next_prev_action = self._storage[head + n].prev_action_list
        next_candidate = self._storage[head + n].action_candidate_list
        next_graph_triplets = self._storage[head + n].graph_triplets

        tmp = next_final - head + 1 if self.accumulate_reward_from_final else n + 1

        rewards_up_to_next_final = [self.discount_gamma_game_reward ** i * self._storage[head + i].reward for i in range(tmp)]
        reward = torch.sum(torch.stack(rewards_up_to_next_final))

        graph_rewards_up_to_next_final = [self.discount_gamma_graph_reward ** i * self._storage[head + i].graph_reward for i in range(tmp)]
        graph_reward = torch.sum(torch.stack(graph_rewards_up_to_next_final))

        count_rewards_up_to_next_final = [self.discount_gamma_count_reward ** i * self._storage[head + i].count_reward for i in range(tmp)]
        count_reward = torch.sum(torch.stack(count_rewards_up_to_next_final))

        return (obs, prev_action, candidate, chosen_indices, graph_triplets, reward + graph_reward + count_reward, next_obs, next_prev_action, next_candidate, next_graph_triplets)

    def _encode_sample(self, idxes, ns):
        actual_indices, actual_ns = [], []
        obs, prev_action, candidate, chosen_indices, graph_triplets, reward, next_obs, next_prev_action, next_candidate, next_graph_triplets = [], [], [], [], [], [], [], [], [], []
        for i, n in zip(idxes, ns):
            t = self._get_single_transition(i, n)
            if t is None:
                continue
            actual_indices.append(i)
            actual_ns.append(n)
            obs.append(t[0])
            prev_action.append(t[1])
            candidate.append(t[2])
            chosen_indices.append(t[3])
            graph_triplets.append(t[4])
            reward.append(t[5])
            next_obs.append(t[6])
            next_prev_action.append(t[7])
            next_candidate.append(t[8])
            next_graph_triplets.append(t[9])
        if len(actual_indices) == 0:
            return None
        chosen_indices = np.array(chosen_indices)  # batch
        reward = torch.stack(reward, 0)  # batch
        actual_ns = np.array(actual_ns)

        return [obs, prev_action, candidate, chosen_indices, graph_triplets, reward, next_obs, next_prev_action, next_candidate, next_graph_triplets, actual_indices, actual_ns]

    def sample(self, batch_size, beta=0, multi_step=1):

        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        # sample n
        ns = self.rng.randint(1, multi_step + 1, size=batch_size)
        encoded_sample = self._encode_sample(idxes, ns)
        if encoded_sample is None:
            return None
        actual_indices = encoded_sample[-2]
        for idx in actual_indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return encoded_sample + [weights]

    def _get_single_sequence_transition(self, idx, sample_history_length):
        assert sample_history_length > 0
        head = idx
        # if n is 1, then head can't be is_final
        if sample_history_length == 1:
            if self._storage[head].is_final:
                return None
        #  if n > 1, then all except tail can't be is_final
        else:
            if np.any([item.is_final for item in self._storage[head: head + sample_history_length]]):
                return None

        next_final = self.get_next_final_pos(self._storage, head)
        if next_final is None:
            return None

        # all good
        res = []
        for m in range(sample_history_length):
            obs = self._storage[head + m].observation_list
            candidate = self._storage[head + m].action_candidate_list
            chosen_indices = self._storage[head + m].chosen_indices
            graph_triplets = self._storage[head + m].graph_triplets

            next_obs = self._storage[head + m + 1].observation_list
            next_candidate = self._storage[head + m + 1].action_candidate_list
            next_graph_triplets = self._storage[head + m + 1].graph_triplets

            tmp = next_final - (head + m) + 1 if self.accumulate_reward_from_final else 1

            rewards_up_to_next_final = [self.discount_gamma_game_reward ** i * self._storage[head + m + i].reward for i in range(tmp)]
            reward = torch.sum(torch.stack(rewards_up_to_next_final))

            graph_rewards_up_to_next_final = [self.discount_gamma_graph_reward ** i * self._storage[head + m + i].graph_reward for i in range(tmp)]
            graph_reward = torch.sum(torch.stack(graph_rewards_up_to_next_final))

            count_rewards_up_to_next_final = [self.discount_gamma_count_reward ** i * self._storage[head + m + i].count_reward for i in range(tmp)]
            count_reward = torch.sum(torch.stack(count_rewards_up_to_next_final))

            res.append([obs, candidate, chosen_indices, graph_triplets, reward + graph_reward + count_reward, next_obs, next_candidate, next_graph_triplets])
        return res

    def _encode_sample_sequence(self, idxes, sample_history_length):
        assert sample_history_length > 0
        res = []
        for _ in range(sample_history_length):
            tmp = []
            for i in range(8):
                tmp.append([])
            res.append(tmp)

        actual_indices = []
        # obs, candidate, chosen_indices, graph_triplets, reward, next_obs, next_candidate, next_graph_triplets
        for i in idxes:
            t = self._get_single_sequence_transition(i, sample_history_length)
            if t is None:
                continue
            actual_indices.append(i)
            for step in range(sample_history_length):
                t_s = t[step]
                res[step][0].append(t_s[0])
                res[step][1].append(t_s[1])
                res[step][2].append(t_s[2])
                res[step][3].append(t_s[3])
                res[step][4].append(t_s[4])
                res[step][5].append(t_s[5])
                res[step][6].append(t_s[6])
                res[step][7].append(t_s[7])

        if len(actual_indices) == 0:
            return None
        for i in range(sample_history_length):
            res[i][2] = np.array(res[i][2])  # batch
            res[i][4] = torch.stack(res[i][4], 0)  # batch

        return res + [actual_indices]

    def sample_sequence(self, batch_size, beta=0, sample_history_length=1):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        res_weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        encoded_sample = self._encode_sample_sequence(idxes, sample_history_length)
        if encoded_sample is None:
            return None
        actual_indices = encoded_sample[-1]
        for _h in range(sample_history_length):
            tmp_weights = []
            for idx in actual_indices:
                p_sample = self._it_sum[idx + _h] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                tmp_weights.append(weight / max_weight)
            tmp_weights = np.array(tmp_weights)
            res_weights.append(tmp_weights)

        return encoded_sample + [res_weights]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = self.rng.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            if priority > 0:
                assert 0 <= idx < len(self._storage)
                self._it_sum[idx] = priority ** self._alpha
                self._it_min[idx] = priority ** self._alpha
                self._max_priority = max(self._max_priority, priority)
            else:
                print("something wrong with priority: ", str(priority))
                return False
        return True

    def avg_rewards(self):
        if len(self._storage) == 0:
            return 0.0
        rewards = [self._storage[i].reward for i in range(len(self._storage))]
        return to_np(torch.mean(torch.stack(rewards)))
