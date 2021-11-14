import os
import random
import copy
import codecs
import spacy
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from textworld import EnvInfos

import dqn_memory_priortized_replay_buffer
from model import KG_Manipulation
from generic import to_np, to_pt, _words_to_ids, _word_to_id, pad_sequences, update_graph_triplets, preproc, max_len, ez_gather_dim_1
from generic import sort_target_commands, process_facts, serialize_facts, gen_graph_commands, process_fully_obs_facts
from generic import generate_labels_for_ap, generate_labels_for_sp, LinearSchedule
from layers import NegativeLogLoss, compute_mask, masked_mean


class Agent:
    def __init__(self, config):
        self.mode = "train"
        self.config = config
        print(self.config)
        self.load_config()

        self.online_net = KG_Manipulation(config=self.config, word_vocab=self.word_vocab, node_vocab=self.node_vocab, relation_vocab=self.relation_vocab)
        self.online_net.train()
        if self.use_cuda:
            self.online_net.cuda()

        if self.task == "rl":
            self.target_net = KG_Manipulation(config=self.config, word_vocab=self.word_vocab, node_vocab=self.node_vocab, relation_vocab=self.relation_vocab)
            self.pretrained_graph_generation_net = KG_Manipulation(config=self.config, word_vocab=self.word_vocab, node_vocab=self.node_vocab, relation_vocab=self.relation_vocab)
            self.target_net.train()
            self.pretrained_graph_generation_net.eval()
            self.update_target_net()
            for param in self.target_net.parameters():
                param.requires_grad = False
            for param in self.pretrained_graph_generation_net.parameters():
                param.requires_grad = False
            if self.use_cuda:
                self.target_net.cuda()
                self.pretrained_graph_generation_net.cuda()
        else:
            self.target_net, self.pretrained_graph_generation_net = None, None

        # exclude some parameters from optimizer
        param_frozen_list = [] # should be changed into torch.nn.ParameterList()
        param_active_list = [] # should be changed into torch.nn.ParameterList()
        for k, v in self.online_net.named_parameters():
            keep_this = True
            for keyword in self.fix_parameters_keywords:
                if keyword in k:
                    param_frozen_list.append(v)
                    keep_this = False
                    break
            if keep_this:
                param_active_list.append(v)

        param_frozen_list = torch.nn.ParameterList(param_frozen_list)
        param_active_list = torch.nn.ParameterList(param_active_list)

        # optimizer
        if self.step_rule == 'adam':
            self.optimizer = torch.optim.Adam([{'params': param_frozen_list, 'lr': 0.0},
                                               {'params': param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                              lr=self.config['general']['training']['optimizer']['learning_rate'])
        elif self.step_rule == 'radam':
            from radam import RAdam
            self.optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                    {'params': param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                   lr=self.config['general']['training']['optimizer']['learning_rate'])
        else:
            raise NotImplementedError

    def load_config(self):
        self.real_valued_graph = self.config['general']['model']['real_valued_graph']
        self.task = self.config['general']['task']
        # word vocab
        self.word_vocab = []
        with codecs.open("./vocabularies/word_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        # node vocab
        self.node_vocab = []
        with codecs.open("./vocabularies/node_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.node_vocab.append(line.strip().lower())
        self.node2id = {}
        for i, w in enumerate(self.node_vocab):
            self.node2id[w] = i
        # relation vocab
        self.relation_vocab = []
        with codecs.open("./vocabularies/relation_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.relation_vocab.append(line.strip().lower())
        self.origin_relation_number = len(self.relation_vocab)
        # add reverse relations
        for i in range(self.origin_relation_number):
            self.relation_vocab.append(self.relation_vocab[i] + "_reverse")
        if not self.real_valued_graph:
            # add self relation
            self.relation_vocab += ["self"]
        self.relation2id = {}
        for i, w in enumerate(self.relation_vocab):
            self.relation2id[w] = i

        self.step_rule = self.config['general']['training']['optimizer']['step_rule']
        self.init_learning_rate = self.config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']
        self.learning_rate_warmup_until = self.config['general']['training']['optimizer']['learning_rate_warmup_until']
        self.fix_parameters_keywords = list(set(self.config['general']['training']['fix_parameters_keywords']))
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training']['smoothing_eps']
        self.patience = self.config['general']['training']['patience']

        self.run_eval = self.config['general']['evaluate']['run_eval']
        self.eval_g_belief = self.config['general']['evaluate']['g_belief']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']
        self.max_target_length = self.config['general']['evaluate']['max_target_length']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.experiment_tag = self.config['general']['checkpoint']['experiment_tag']
        self.save_frequency = self.config['general']['checkpoint']['save_frequency']
        self.report_frequency = self.config['general']['checkpoint']['report_frequency']
        self.load_pretrained = self.config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint']['load_from_tag']
        self.load_graph_generation_model_from_tag = self.config['general']['checkpoint']['load_graph_generation_model_from_tag']
        self.load_parameter_keywords = list(set(self.config['general']['checkpoint']['load_parameter_keywords']))

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.backprop_frequency = self.config['obs_gen']['backprop_frequency']

        # AP specific
        self.ap_k_way_classification = self.config['ap']['k_way_classification']

        # SP specific
        self.sp_k_way_classification = self.config['sp']['k_way_classification']

        # DGI specific
        self.sample_bias_positive = self.config['dgi']['sample_bias_positive']
        self.sample_bias_negative = self.config['dgi']['sample_bias_negative']

        # RL specific
        self.fully_observable_graph = self.config['rl']['fully_observable_graph']
        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.epsilon_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes, initial_p=self.epsilon_anneal_from, final_p=self.epsilon_anneal_to)
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0
        # drqn
        self.replay_sample_history_length = self.config['rl']['replay']['replay_sample_history_length']
        self.replay_sample_update_from = self.config['rl']['replay']['replay_sample_update_from']
        # replay buffer and updates
        self.buffer_reward_threshold = self.config['rl']['replay']['buffer_reward_threshold']
        self.prioritized_replay_beta = self.config['rl']['replay']['prioritized_replay_beta']
        self.beta_scheduler = LinearSchedule(schedule_timesteps=self.max_episode, initial_p=self.prioritized_replay_beta, final_p=1.0)

        self.accumulate_reward_from_final = self.config['rl']['replay']['accumulate_reward_from_final']
        self.prioritized_replay_eps = self.config['rl']['replay']['prioritized_replay_eps']
        self.count_reward_lambda = self.config['rl']['replay']['count_reward_lambda']
        self.discount_gamma_count_reward = self.config['rl']['replay']['discount_gamma_count_reward']
        self.graph_reward_lambda = self.config['rl']['replay']['graph_reward_lambda']
        self.graph_reward_type = self.config['rl']['replay']['graph_reward_type']
        self.discount_gamma_graph_reward = self.config['rl']['replay']['discount_gamma_graph_reward']
        self.discount_gamma_game_reward = self.config['rl']['replay']['discount_gamma_game_reward']
        self.replay_batch_size = self.config['rl']['replay']['replay_batch_size']
        self.dqn_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(self.config['rl']['replay']['replay_memory_capacity'],
                                                                                      priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                                                                                      discount_gamma_game_reward=self.discount_gamma_game_reward,
                                                                                      discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                                                                                      discount_gamma_count_reward=self.discount_gamma_count_reward,
                                                                                      accumulate_reward_from_final=self.accumulate_reward_from_final,
                                                                                      seed=self.config['general']['random_seed'])
        self.update_per_k_game_steps = self.config['rl']['replay']['update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        # input in rl training
        self.enable_recurrent_memory = self.config['rl']['model']['enable_recurrent_memory']
        self.enable_graph_input = self.config['rl']['model']['enable_graph_input']
        self.enable_text_input = self.config['rl']['model']['enable_text_input']
        assert self.enable_graph_input or self.enable_text_input
        # rl train and eval
        self.max_nb_steps_per_episode = self.config['rl']['training']['max_nb_steps_per_episode']
        self.learn_start_from_this_episode = self.config['rl']['training']['learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training']['target_net_update_frequency']
        self.use_negative_reward = self.config['rl']['training']['use_negative_reward']
        self.eval_max_nb_steps_per_episode = self.config['rl']['evaluate']['max_nb_steps_per_episode']

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def update_target_net(self):
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.online_net.zero_noise()
            if self.target_net is not None:
                self.target_net.zero_noise()
            if self.pretrained_graph_generation_net is not None:
                self.pretrained_graph_generation_net.zero_noise()

    def load_pretrained_graph_generation_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading pre-trained graph generation model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')
            try:
                self.pretrained_graph_generation_net.load_state_dict(pretrained_dict)
            except:
                # graph generation net
                model_dict = self.pretrained_graph_generation_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.pretrained_graph_generation_net.load_state_dict(model_dict)
                print("WARNING... Model dict is different with pretrained dict. I'm loading only the parameters with same labels now. Make sure you really want this...")
                print("The loaded parameters are:")
                keys = [key for key in pretrained_dict]
                print(", ".join(keys))
                print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def load_pretrained_model(self, load_from, load_partial_graph=True):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def select_additional_infos(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = True
        request_infos.location = True
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def select_additional_infos_lite(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = False
        request_infos.location = False
        request_infos.facts = False
        request_infos.last_action = False
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def init(self):
        pass

    def get_word_input(self, input_strings):
        word_list = [item.split() for item in input_strings]
        word_id_list = [_words_to_ids(tokens, self.word2id) for tokens in word_list]
        input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def get_graph_adjacency_matrix(self, triplets):
        adj = np.zeros((len(triplets), len(self.relation_vocab), len(self.node_vocab), len(self.node_vocab)), dtype="float32")
        for b in range(len(triplets)):
            node_exists = set()
            for t in triplets[b]:
                node1, node2, relation = t
                assert node1 in self.node_vocab, node1 + " is not in node vocab"
                assert node2 in self.node_vocab, node2 + " is not in node vocab"
                assert relation in self.relation_vocab, relation + " is not in relation vocab"
                node1_id, node2_id, relation_id = _word_to_id(node1, self.node2id), _word_to_id(node2, self.node2id), _word_to_id(relation, self.relation2id)
                adj[b][relation_id][node1_id][node2_id] = 1.0
                adj[b][relation_id + self.origin_relation_number][node2_id][node1_id] = 1.0
                node_exists.add(node1_id)
                node_exists.add(node2_id)
            # self relation
            for node_id in list(node_exists):
                adj[b, -1, node_id, node_id] = 1.0
        adj = to_pt(adj, self.use_cuda, type='float')
        return adj

    def get_graph_node_name_input(self):
        res = copy.copy(self.node_vocab)
        input_node_name = self.get_word_input(res)  # num_node x words
        return input_node_name

    def get_graph_relation_name_input(self):
        res = copy.copy(self.relation_vocab)
        res = [item.replace("_", " ") for item in res]
        input_relation_name = self.get_word_input(res)  # num_node x words
        return input_relation_name

    def get_action_candidate_list_input(self, action_candidate_list):
        # action_candidate_list (list): batch x num_candidate of strings
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        input_action_candidate_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(action_candidate_list[i])
            input_action_candidate_list.append(word_level)
        max_word_num = max([item.size(1) for item in input_action_candidate_list])

        input_action_candidate = np.zeros((batch_size, max_num_candidate, max_word_num))
        input_action_candidate = to_pt(input_action_candidate, self.use_cuda, type="long")
        for i in range(batch_size):
            input_action_candidate[i, :input_action_candidate_list[i].size(0), :input_action_candidate_list[i].size(1)] = input_action_candidate_list[i]

        return input_action_candidate

    def choose_model(self, use_model="online"):
        if self.task != "rl":
            return self.online_net
        if use_model == "online":
            model = self.online_net
        elif use_model == "target":
            model = self.target_net
        elif use_model == "pretrained_graph_generation":
            model = self.pretrained_graph_generation_net
        else:
            raise NotImplementedError
        return model

    def encode_graph(self, graph_input, use_model):
        model = self.choose_model(use_model)
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        if isinstance(graph_input, list):
            adjacency_matrix = self.get_graph_adjacency_matrix(graph_input)
        elif isinstance(graph_input, torch.Tensor):
            adjacency_matrix = graph_input
        else:
            raise NotImplementedError
        node_encoding_sequence, node_mask = model.encode_graph(input_node_name, input_relation_name, adjacency_matrix)
        return node_encoding_sequence, node_mask

    def encode_text(self, observation_strings, use_model):
        model = self.choose_model(use_model)
        input_obs = self.get_word_input(observation_strings)
        # encode
        obs_encoding_sequence, obs_mask = model.encode_text(input_obs)
        return obs_encoding_sequence, obs_mask

    ##################################
    ## rl with unsupervised graph
    ##################################

    def hidden_to_adjacency_matrix(self, hidden, batch_size, use_model):
        model = self.choose_model(use_model)
        num_node = len(self.node_vocab)
        num_relation = len(self.relation_vocab)
        if hidden is None:
            adjacency_matrix = torch.zeros(batch_size, num_relation, num_node, num_node)
            if self.use_cuda:
                adjacency_matrix = adjacency_matrix.cuda()
        else:
            adjacency_matrix = torch.tanh(model.obs_gen_linear_2(F.relu(model.obs_gen_linear_1(hidden)))).view(batch_size, int(num_relation / 2), num_node, num_node)
            adjacency_matrix = adjacency_matrix.repeat(1, 2, 1, 1)
            for i in range(int(num_relation / 2)):
                adjacency_matrix[:, int(num_relation / 2) + i] = adjacency_matrix[:, i].permute(0, 2, 1)
        return adjacency_matrix

    def generate_adjacency_matrix_for_rl(self, observation_strings, prev_action_strings, h_t_minus_one):
        with torch.no_grad():
            if h_t_minus_one is not None:
                h_t_minus_one = h_t_minus_one.detach()
            # TE-encode
            input_obs = self.get_word_input(observation_strings)
            prev_action_word_ids = self.get_word_input(prev_action_strings)
            prev_action_encoding_sequence, prev_action_mask =  self.pretrained_graph_generation_net.encode_text_for_pretraining_tasks(prev_action_word_ids)
            obs_encoding_sequence, obs_mask =  self.pretrained_graph_generation_net.encode_text_for_pretraining_tasks(input_obs)
            prev_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t_minus_one, batch_size=len(observation_strings), use_model="pretrained_graph_generation")
            node_encoding_sequence, node_mask = self.encode_graph(prev_adjacency_matrix, use_model="pretrained_graph_generation")

            h_ag = self.pretrained_graph_generation_net.obs_gen_attention(prev_action_encoding_sequence, node_encoding_sequence, prev_action_mask, node_mask)
            h_ga = self.pretrained_graph_generation_net.obs_gen_attention(node_encoding_sequence, prev_action_encoding_sequence, node_mask, prev_action_mask)

            h_ag = self.pretrained_graph_generation_net.obs_gen_attention_prj(h_ag)
            h_ga = self.pretrained_graph_generation_net.obs_gen_attention_prj(h_ga)

            h_og = self.pretrained_graph_generation_net.obs_gen_attention(obs_encoding_sequence, node_encoding_sequence, obs_mask, node_mask)
            h_go = self.pretrained_graph_generation_net.obs_gen_attention(node_encoding_sequence, obs_encoding_sequence, node_mask, obs_mask)

            h_og = self.pretrained_graph_generation_net.obs_gen_attention_prj(h_og) # bs X len X block_hidden_dim
            h_go = self.pretrained_graph_generation_net.obs_gen_attention_prj(h_go) # bs X len X block_hidden_dim

            ave_h_go = masked_mean(h_go, m=node_mask, dim=1)
            ave_h_og = masked_mean(h_og, m=obs_mask, dim=1)
            ave_h_ga = masked_mean(h_ga, m=node_mask, dim=1)
            ave_h_ag = masked_mean(h_ag, m=prev_action_mask, dim=1)

            rnn_input = self.pretrained_graph_generation_net.obs_gen_attention_to_rnn_input(torch.cat([ave_h_go, ave_h_og, ave_h_ga, ave_h_ag], dim=1))  # batch x block_hidden_dim
            rnn_input = torch.tanh(rnn_input)  # batch x block_hidden_dim
            h_t = self.pretrained_graph_generation_net.obs_gen_graph_rnncell(rnn_input, h_t_minus_one) if h_t_minus_one is not None else self.pretrained_graph_generation_net.obs_gen_graph_rnncell(rnn_input)  # both batch x block_hidden_dim
            current_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t, batch_size=len(observation_strings), use_model="pretrained_graph_generation")

            return current_adjacency_matrix.detach(), h_t.detach()

    ##################################
    # observation generation specific
    ##################################

    def get_observation_infomax_logits(self, observation_strings, prev_action_strings, corrupted_observation_strings, h_t_minus_one):
        # TE-encode
        input_obs = self.get_word_input(observation_strings)
        prev_action_word_ids = self.get_word_input(prev_action_strings)
        prev_action_encoding_sequence, prev_action_mask = self.online_net.encode_text_for_pretraining_tasks(prev_action_word_ids)
        obs_encoding_sequence, obs_mask = self.online_net.encode_text_for_pretraining_tasks(input_obs)
        # TE-corrupted encode
        corrupted_input_obs = self.get_word_input(corrupted_observation_strings)
        corrupted_obs_encoding_sequence, corrupted_obs_mask = self.online_net.encode_text_for_pretraining_tasks(corrupted_input_obs)
        # adj matrix
        prev_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t_minus_one, batch_size=len(observation_strings), use_model="online")
        node_encoding_sequence, node_mask = self.encode_graph(prev_adjacency_matrix, use_model="online")

        h_ag = self.online_net.obs_gen_attention(prev_action_encoding_sequence, node_encoding_sequence, prev_action_mask, node_mask)
        h_ga = self.online_net.obs_gen_attention(node_encoding_sequence, prev_action_encoding_sequence, node_mask, prev_action_mask)
        h_ag = self.online_net.obs_gen_attention_prj(h_ag)
        h_ga = self.online_net.obs_gen_attention_prj(h_ga)

        h_og = self.online_net.obs_gen_attention(obs_encoding_sequence, node_encoding_sequence, obs_mask, node_mask)
        h_go = self.online_net.obs_gen_attention(node_encoding_sequence, obs_encoding_sequence, node_mask, obs_mask)
        h_og = self.online_net.obs_gen_attention_prj(h_og) # bs X len X block_hidden_dim
        h_go = self.online_net.obs_gen_attention_prj(h_go) # bs X len X block_hidden_dim

        ave_h_go = masked_mean(h_go, m=node_mask, dim=1)
        ave_h_og = masked_mean(h_og, m=obs_mask, dim=1)
        ave_h_ga = masked_mean(h_ga, m=node_mask, dim=1)
        ave_h_ag = masked_mean(h_ag, m=prev_action_mask, dim=1)

        rnn_input = self.online_net.obs_gen_attention_to_rnn_input(torch.cat([ave_h_go, ave_h_og, ave_h_ga, ave_h_ag], dim=1))  # batch x block_hidden_dim
        rnn_input = torch.tanh(rnn_input)  # batch x block_hidden_dim
        h_t = self.online_net.obs_gen_graph_rnncell(rnn_input, h_t_minus_one) if h_t_minus_one is not None else self.online_net.obs_gen_graph_rnncell(rnn_input)  # both batch x block_hidden_dim
        current_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t, batch_size=len(observation_strings), use_model="online")
        new_node_encoding_sequence, new_node_mask = self.encode_graph(current_adjacency_matrix, use_model="online")

        h_ag2 = self.online_net.obs_gen_attention(prev_action_encoding_sequence, new_node_encoding_sequence, prev_action_mask, new_node_mask)
        h_ga2 = self.online_net.obs_gen_attention(new_node_encoding_sequence, prev_action_encoding_sequence, new_node_mask, prev_action_mask)
        h_ag2 = self.online_net.obs_gen_attention_prj(h_ag2)
        h_ga2 = self.online_net.obs_gen_attention_prj(h_ga2)
        ave_h_ag2 = masked_mean(h_ag2, m=prev_action_mask, dim=1)
        ave_h_ga2 = masked_mean(h_ga2, m=new_node_mask, dim=1)
        logits = self.online_net.observation_discriminator(torch.cat([ave_h_ag2, ave_h_ga2], dim=-1), obs_encoding_sequence, obs_mask, corrupted_obs_encoding_sequence, corrupted_obs_mask)
        return logits, h_t

    def get_observation_infomax_loss(self, observation_strings, prev_action_strings, evaluate=False):
        curr_batch_size = len(observation_strings)
        lens = [len(elem) for elem in observation_strings]
        max_len = max(lens)
        episodes_masks = torch.zeros((curr_batch_size, max_len), dtype=torch.float).cuda() if self.use_cuda else torch.zeros((curr_batch_size, max_len), dtype=torch.float)
        for i in range(curr_batch_size):
            episodes_masks[i, :lens[i]] = 1
        episodes_masks = episodes_masks.repeat(2, 1) # repeat for corrupted obs

        observation_strings = [elem + ["<pad>"]*(max_len - len(elem)) for elem in observation_strings]
        prev_action_strings = [elem + ["<pad>"]*(max_len - len(elem)) for elem in prev_action_strings]
        prev_h = None

        last_k_batches_loss = []
        return_losses = []
        return_accuracies = []

        for i in range(max_len):
            current_step_eps_masks = episodes_masks[:, i]
            batch_obs_strings, batch_prev_action_strings, batch_corrupted_obs_strings = [], [], []
            for j in range(curr_batch_size):
                batch_obs_strings.append(observation_strings[j][i])
                if lens[j] == 1 or random.random() > 0.7:
                    random_id_from_batch = random.choice(range(len(observation_strings)))
                    batch_corrupted_obs_strings.append(observation_strings[random_id_from_batch][random.choice(range(lens[random_id_from_batch]))])
                    while(batch_corrupted_obs_strings[-1] == batch_obs_strings[-1]):
                        random_id_from_batch = random.choice(range(len(observation_strings)))
                        batch_corrupted_obs_strings[-1] = observation_strings[random_id_from_batch][random.choice(range(lens[random_id_from_batch]))]
                else:
                    batch_corrupted_obs_strings.append(observation_strings[j][random.choice(range(lens[j]))])
                    while(batch_corrupted_obs_strings[-1] == batch_obs_strings[-1]):
                        batch_corrupted_obs_strings[-1] = observation_strings[j][random.choice(range(lens[j]))]
                batch_prev_action_strings.append(prev_action_strings[j][i])
            logits, prev_h = self.get_observation_infomax_logits(batch_obs_strings, batch_prev_action_strings, batch_corrupted_obs_strings, prev_h)

            # labels
            labels_positive = torch.ones(curr_batch_size) # bs,
            labels_negative = torch.zeros(curr_batch_size) # bs,
            labels = torch.cat([labels_positive, labels_negative]) # bs*2,
            if self.use_cuda:
                labels = labels.cuda()

            loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits.squeeze(1), labels)
            loss = torch.sum(loss * current_step_eps_masks) / torch.sum(current_step_eps_masks)
            return_losses.append(to_np(loss))
            preds = to_np(logits.squeeze(1))
            preds = (preds > 0.5).astype("int32")
            for m in range(curr_batch_size):
                if current_step_eps_masks[m] == 0:
                    continue
                return_accuracies.append(float(preds[m] == 1))
            for m in range(curr_batch_size):
                if current_step_eps_masks[m] == 0:
                    continue
                return_accuracies.append(float(preds[curr_batch_size + m] == 0))
            if evaluate:
                continue
            last_k_batches_loss.append(loss.unsqueeze(0))
            if ((i + 1) % self.backprop_frequency == 0 or i == max_len - 1) and i > 0:
                self.optimizer.zero_grad()
                torch_last_k_batches_loss = torch.cat(last_k_batches_loss)
                ave_k_loss = torch.mean(torch_last_k_batches_loss)
                ave_k_loss.backward()
                self.optimizer.step()
                last_k_batches_loss = []
                prev_h = prev_h.detach()

        return return_losses, return_accuracies

    def observation_generation_teacher_force(self, observation_strings, prev_action_strings, episodes_masks, h_t_minus_one):
        input_observation_strings = [" ".join(["<bos>"] + item.split()) for item in observation_strings]
        output_observation_strings = [" ".join(item.split() + ["<eos>"]) for item in observation_strings]
        ground_truth = self.get_word_input(output_observation_strings)
        # TE-encode
        input_obs = self.get_word_input(observation_strings)
        prev_action_word_ids = self.get_word_input(prev_action_strings)
        prev_action_encoding_sequence, prev_action_mask = self.online_net.encode_text_for_pretraining_tasks(prev_action_word_ids)
        obs_encoding_sequence, obs_mask = self.online_net.encode_text_for_pretraining_tasks(input_obs)
        prev_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t_minus_one, batch_size=len(observation_strings), use_model="online")
        node_encoding_sequence, node_mask = self.encode_graph(prev_adjacency_matrix, use_model="online")

        h_ag = self.online_net.obs_gen_attention(prev_action_encoding_sequence, node_encoding_sequence, prev_action_mask, node_mask)
        h_ga = self.online_net.obs_gen_attention(node_encoding_sequence, prev_action_encoding_sequence, node_mask, prev_action_mask)
        h_ag = self.online_net.obs_gen_attention_prj(h_ag)
        h_ga = self.online_net.obs_gen_attention_prj(h_ga)

        h_og = self.online_net.obs_gen_attention(obs_encoding_sequence, node_encoding_sequence, obs_mask, node_mask)
        h_go = self.online_net.obs_gen_attention(node_encoding_sequence, obs_encoding_sequence, node_mask, obs_mask)
        h_og = self.online_net.obs_gen_attention_prj(h_og) # bs X len X block_hidden_dim
        h_go = self.online_net.obs_gen_attention_prj(h_go) # bs X len X block_hidden_dim

        ave_h_go = masked_mean(h_go, m=node_mask, dim=1)
        ave_h_og = masked_mean(h_og, m=obs_mask, dim=1)
        ave_h_ga = masked_mean(h_ga, m=node_mask, dim=1)
        ave_h_ag = masked_mean(h_ag, m=prev_action_mask, dim=1)

        rnn_input = self.online_net.obs_gen_attention_to_rnn_input(torch.cat([ave_h_go, ave_h_og, ave_h_ga, ave_h_ag], dim=1))  # batch x block_hidden_dim
        rnn_input = torch.tanh(rnn_input)  # batch x block_hidden_dim
        h_t = self.online_net.obs_gen_graph_rnncell(rnn_input, h_t_minus_one) if h_t_minus_one is not None else self.online_net.obs_gen_graph_rnncell(rnn_input)  # both batch x block_hidden_dim

        current_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t, batch_size=len(observation_strings), use_model="online")
        new_node_encoding_sequence, new_node_mask = self.encode_graph(current_adjacency_matrix, use_model="online")

        h_ag2 = self.online_net.obs_gen_attention(prev_action_encoding_sequence, new_node_encoding_sequence, prev_action_mask, new_node_mask)
        h_ga2 = self.online_net.obs_gen_attention(new_node_encoding_sequence, prev_action_encoding_sequence, new_node_mask, prev_action_mask)
        h_ag2 = self.online_net.obs_gen_attention_prj(h_ag2)
        h_ga2 = self.online_net.obs_gen_attention_prj(h_ga2)
        input_target = self.get_word_input(input_observation_strings)
        target_mask = compute_mask(input_target)

        pred = self.online_net.decode_for_obs_gen(input_target, h_ag2, prev_action_mask, h_ga2, new_node_mask)
        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
        loss = torch.sum(batch_loss * episodes_masks) / torch.sum(episodes_masks) # only place where using `episodes_masks`
        return loss, pred * target_mask.unsqueeze(-1), h_t

    def observation_generation_greedy_generation(self, observation_strings, prev_action_strings, episodes_masks, h_t_minus_one=None):
        with torch.no_grad():
            batch_size = len(observation_strings)
            # encode
            input_obs = self.get_word_input(observation_strings)
            prev_action_word_ids = self.get_word_input(prev_action_strings)
            model = self.choose_model("online")

            obs_encoding_sequence, obs_mask = model.encode_text_for_pretraining_tasks(input_obs)
            prev_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t_minus_one, batch_size=len(observation_strings), use_model="online")
            node_encoding_sequence, node_mask = self.encode_graph(prev_adjacency_matrix, use_model="online")
            prev_action_sequence, prev_action_mask = model.encode_text_for_pretraining_tasks(prev_action_word_ids)

            h_ag = self.online_net.obs_gen_attention(prev_action_sequence, node_encoding_sequence, prev_action_mask, node_mask)
            h_ga = self.online_net.obs_gen_attention(node_encoding_sequence, prev_action_sequence, node_mask, prev_action_mask)
            h_ag = self.online_net.obs_gen_attention_prj(h_ag)
            h_ga = self.online_net.obs_gen_attention_prj(h_ga)

            h_og = self.online_net.obs_gen_attention(obs_encoding_sequence, node_encoding_sequence, obs_mask, node_mask)
            h_go = self.online_net.obs_gen_attention(node_encoding_sequence, obs_encoding_sequence, node_mask, obs_mask)
            h_og = self.online_net.obs_gen_attention_prj(h_og) # bs X len X block_hidden_dim
            h_go = self.online_net.obs_gen_attention_prj(h_go) # bs X len X block_hidden_dim

            ave_h_go = masked_mean(h_go, m=node_mask, dim=1)
            ave_h_og = masked_mean(h_og, m=obs_mask, dim=1)
            ave_h_ga = masked_mean(h_ga, m=node_mask, dim=1)
            ave_h_ag = masked_mean(h_ag, m=prev_action_mask, dim=1)

            rnn_input = self.online_net.obs_gen_attention_to_rnn_input(torch.cat([ave_h_go, ave_h_og, ave_h_ga, ave_h_ag], dim=1))  # batch x block_hidden_dim
            rnn_input = torch.tanh(rnn_input)  # batch x block_hidden_dim
            h_t = self.online_net.obs_gen_graph_rnncell(rnn_input, h_t_minus_one) if h_t_minus_one is not None else self.online_net.obs_gen_graph_rnncell(rnn_input)  # both batch x block_hidden_dim
            current_adjacency_matrix = self.hidden_to_adjacency_matrix(h_t, batch_size=len(observation_strings), use_model="online")

            new_node_encoding_sequence, new_node_mask = self.encode_graph(current_adjacency_matrix, use_model="online")
            h_ag2 = self.online_net.obs_gen_attention(prev_action_sequence, new_node_encoding_sequence, prev_action_mask, new_node_mask)
            h_ga2 = self.online_net.obs_gen_attention(new_node_encoding_sequence, prev_action_sequence, new_node_mask, prev_action_mask)
            h_ag2 = self.online_net.obs_gen_attention_prj(h_ag2)
            h_ga2 = self.online_net.obs_gen_attention_prj(h_ga2)

            input_target_token_list = [["<bos>"] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = self.get_word_input([" ".join(item) for item in input_target_token_list])
                pred = model.decode_for_obs_gen(input_target, h_ag2, prev_action_mask, h_ga2, new_node_mask)  # batch x time x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [self.word_vocab[pred[b]]] if eos[b] == 0 else []
                    input_target_token_list[b] = input_target_token_list[b] + new_stuff
                    if pred[b] == self.word2id["<eos>"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            return [" ".join(item[1:]) for item in input_target_token_list], h_t

    ##################################
    # command generation specific
    ##################################

    # command generation stuff (supervised learning)
    def command_generation_teacher_force(self, observation_strings, triplets, target_strings):
        input_target_strings = [" ".join(["<bos>"] + item.split()) for item in target_strings]
        output_target_strings = [" ".join(item.split() + ["<eos>"]) for item in target_strings]
        ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
        # encode
        input_obs = self.get_word_input(observation_strings)
        obs_encoding_sequence, obs_mask = self.online_net.encode_text_for_pretraining_tasks(input_obs)
        node_encoding_sequence, node_mask = self.encode_graph(triplets, use_model="online")

        h_og = self.online_net.cmd_gen_attention(obs_encoding_sequence, node_encoding_sequence, obs_mask, node_mask)
        h_go = self.online_net.cmd_gen_attention(node_encoding_sequence, obs_encoding_sequence, node_mask, obs_mask)
        h_og = self.online_net.cmd_gen_attention_prj(h_og)
        h_go = self.online_net.cmd_gen_attention_prj(h_go)

        # step 2, supervised
        input_target = self.get_word_input(input_target_strings)
        target_mask = compute_mask(input_target)
        pred = self.online_net.decode(input_target, h_og, obs_mask, h_go, node_mask, input_obs)  # batch x target_length x vocab

        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
        loss = torch.mean(batch_loss)

        if loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(pred), to_np(loss)

    def command_generation_greedy_generation(self, observation_strings, triplets):
        with torch.no_grad():
            batch_size = len(observation_strings)
            # encode
            input_obs = self.get_word_input(observation_strings)
            model = self.choose_model("pretrained_graph_generation")
            obs_encoding_sequence, obs_mask = model.encode_text_for_pretraining_tasks(input_obs)
            node_encoding_sequence, node_mask = self.encode_graph(triplets, use_model="pretrained_graph_generation" if self.task == "rl" else "online")

            h_og = model.cmd_gen_attention(obs_encoding_sequence, node_encoding_sequence, obs_mask, node_mask)
            h_go = model.cmd_gen_attention(node_encoding_sequence, obs_encoding_sequence, node_mask, obs_mask)
            h_og = model.cmd_gen_attention_prj(h_og)
            h_go = model.cmd_gen_attention_prj(h_go)

            # step 2, greedy generation
            # decode
            input_target_token_list = [["<bos>"] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = self.get_word_input([" ".join(item) for item in input_target_token_list])
                pred = model.decode(input_target, h_og, obs_mask, h_go, node_mask, input_obs)  # batch x time x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [self.word_vocab[pred[b]]] if eos[b] == 0 else []
                    input_target_token_list[b] = input_target_token_list[b] + new_stuff
                    if pred[b] == self.word2id["<eos>"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            return [" ".join(item[1:]) for item in input_target_token_list]

    ##################################
    # action prediction specific
    ##################################

    # action prediction stuff (supervised learning)
    def get_action_prediction_logits(self, current_triplets, previous_triplets, target_action, action_candidates):

        h_g, node_mask = self.encode_graph(current_triplets, use_model="online")
        prev_h_g, prev_node_mask = self.encode_graph(previous_triplets, use_model="online")
        graph_mask = torch.gt(node_mask + prev_node_mask, 0.0).float()  # tmp_batch x num_node

        labels, action_candidate_list = generate_labels_for_ap(target_action, action_candidates, k_way_classification=self.ap_k_way_classification)
        input_action_candidate = self.get_action_candidate_list_input(action_candidate_list)  # batch x num_cand x cand_length

        batch_size = len(action_candidate_list)
        num_candidate, candidate_len = input_action_candidate.size(1), input_action_candidate.size(2)
        np_labels = pad_sequences(labels)
        pt_labels = to_pt(np_labels, self.use_cuda, type="long")

        cand_mask = np.zeros((batch_size, num_candidate), dtype="float32")
        for b in range(batch_size):
            cand_mask[b, :len(action_candidate_list[b])] = 1.0
        cand_mask = to_pt(cand_mask, self.use_cuda, type='float')

        # attention on [g_{t}] <-> [g_{t+1}]
        attended_h_g = self.online_net.ap_attention_prj(self.online_net.ap_attention(h_g, prev_h_g, node_mask, prev_node_mask))
        ave_attended_h_g = masked_mean(attended_h_g, graph_mask, dim=1).unsqueeze(1)
        prev_attended_h_g = self.online_net.ap_attention_prj(self.online_net.ap_attention(prev_h_g, h_g, prev_node_mask, node_mask))
        ave_prev_attended_h_g = masked_mean(prev_attended_h_g, graph_mask, dim=1).unsqueeze(1)

        global_g = ave_attended_h_g.expand(-1, num_candidate, -1)  # batch x num_cand x hid
        prev_global_g = ave_prev_attended_h_g.expand(-1, num_candidate, -1)  # batch x num_cand x hid
        global_g = global_g * cand_mask.unsqueeze(-1)  # batch x num_cand x hid
        prev_global_g = prev_global_g * cand_mask.unsqueeze(-1)  # batch x num_cand x hid

        cand_encoding = []
        input_action_candidate = input_action_candidate.view(batch_size * num_candidate, candidate_len)
        tmp_batch_size = self.batch_size if self.mode == "train" else self.eval_batch_size
        n_tmp_batches = (input_action_candidate.size(0) + tmp_batch_size - 1) // tmp_batch_size
        for tmp_batch_id in range(n_tmp_batches):
            tmp_batch = input_action_candidate[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]
            tmp_cand_encoding_sequence, tmp_cand_mask = self.online_net.encode_text_for_pretraining_tasks(tmp_batch)
            tmp_cand_encoding = masked_mean(tmp_cand_encoding_sequence, tmp_cand_mask, dim=1)
            cand_encoding.append(tmp_cand_encoding)
        cand_encoding = torch.cat(cand_encoding, 0)  # batch_size*num_candidate x hid
        cand_encoding = cand_encoding.view(batch_size, num_candidate, -1)  # batch_size x num_candidate x hid
        cand_encoding = cand_encoding * cand_mask.unsqueeze(-1)  # batch x num_candidate x hid

        apmlp_input = torch.cat([global_g, prev_global_g, cand_encoding], -1) # batch x num_action, hid * 3
        cand_mask_squared = torch.bmm(cand_mask.unsqueeze(-1), cand_mask.unsqueeze(1))  # batch x num_action x num_action
        ap_ret, _ = self.online_net.ap_self_attention(apmlp_input, cand_mask_squared, apmlp_input, apmlp_input)  # batch x num_actions x hid*3
        ap_ret = ap_ret * cand_mask.unsqueeze(-1)
        ap_ret = self.online_net.ap_linear_1(ap_ret)  # batch x num_action x hid
        ap_ret = torch.relu(ap_ret)
        ap_ret = ap_ret * cand_mask.unsqueeze(-1)
        ap_ret = self.online_net.ap_linear_2(ap_ret).squeeze(-1)  # batch x num_action
        ap_ret = ap_ret * cand_mask

        # masked softmax and NLL loss with mask
        ap_ret = ap_ret.masked_fill((1.0 - cand_mask).bool(), float('-inf'))
        loss = torch.nn.CrossEntropyLoss()(ap_ret, torch.argmax(pt_labels, -1))

        return loss, to_np(ap_ret), np_labels, action_candidate_list

    ##################################
    # state prediction specific
    ##################################

    # state prediction stuff (supervised learning)
    def get_state_prediction_logits(self, previous_triplets, action, target_triplets, graph_candidate_triplets):

        labels, graph_candidate_triplets_list = generate_labels_for_sp(target_triplets, graph_candidate_triplets, k_way_classification=self.sp_k_way_classification)
        np_labels = pad_sequences(labels)
        pt_labels = to_pt(np_labels, self.use_cuda, type='long')
        batch_size, num_candidates = np_labels.shape[0], np_labels.shape[1]

        prev_h_g, prev_node_mask = self.encode_graph(previous_triplets, use_model="online")  # batch x num_node x hid
        input_action = self.get_word_input(action)  # batch x action_len
        action_encoding_sequence, action_mask = self.online_net.encode_text_for_pretraining_tasks(input_action)  # batch x action_len x hid
        action_encoding = masked_mean(action_encoding_sequence, action_mask, dim=1)  # .unsqueeze(1).expand(-1, num_candidates, -1)  # batch x num_cand x hid

        cand_mask = np.zeros((batch_size, num_candidates), dtype="float32")
        for b in range(batch_size):
            cand_mask[b, :len(graph_candidate_triplets_list[b])] = 1.0
        cand_mask = to_pt(cand_mask, self.use_cuda, type='float')

        squeezed_candidate_list, from_which_original_batch = [], []
        for b in range(batch_size):
            squeezed_candidate_list += graph_candidate_triplets_list[b]
            for _ in range(len(graph_candidate_triplets_list[b])):
                from_which_original_batch.append(b)

        tmp_batch_size = self.batch_size if self.mode == "train" else self.eval_batch_size
        n_tmp_batches = (len(squeezed_candidate_list) + tmp_batch_size - 1) // tmp_batch_size
        list_avg_graph_representation = []
        for b in range(batch_size):
            list_avg_graph_representation.append([])
        for tmp_batch_id in range(n_tmp_batches):
            tmp_batch_cand = squeezed_candidate_list[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]  # tmp_batch of graphs
            tmp_batch_from = from_which_original_batch[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]
            from_dict = {}
            for item in tmp_batch_from:
                if item not in from_dict:
                    from_dict[item] = 0
                from_dict[item] += 1

            graph_candidates_h_g, graph_candidates_mask = self.encode_graph(tmp_batch_cand, use_model="online")  # tmp_batch x num_node x hid, tmp_batch x num_node
            exp_prev_h_g, exp_prev_node_mask = [], []
            action_rep = []
            for which in sorted(from_dict):
                exp_prev_h_g.append(prev_h_g[which].unsqueeze(0).expand(from_dict[which], -1, -1))
                exp_prev_node_mask.append(prev_node_mask[which].unsqueeze(0).expand(from_dict[which], -1))
                action_rep.append(action_encoding[which].unsqueeze(0).expand(from_dict[which], -1))
            exp_prev_h_g, exp_prev_node_mask = torch.cat(exp_prev_h_g, 0), torch.cat(exp_prev_node_mask, 0)  # tmp_batch x num_node x hid, tmp_batch x num_node
            action_rep = torch.cat(action_rep, 0).unsqueeze(1).expand(-1, exp_prev_h_g.size(1), -1)  # tmp_batch x num_node x hid

            attended_h_g_cand = self.online_net.sp_attention(exp_prev_h_g, graph_candidates_h_g, exp_prev_node_mask, graph_candidates_mask)
            attended_h_g_cand = self.online_net.sp_attention_prj(attended_h_g_cand)  # tmp_batch x num_node x hid

            attended_cand_h_g = self.online_net.sp_attention(graph_candidates_h_g, exp_prev_h_g, graph_candidates_mask, exp_prev_node_mask)
            attended_cand_h_g = self.online_net.sp_attention_prj(attended_cand_h_g)  # tmp_batch x num_node x hid

            graph_rep = torch.cat([attended_h_g_cand, attended_cand_h_g, action_rep], -1)  # tmp_batch x num_node x hid*3
            graph_mask = torch.gt(exp_prev_node_mask + graph_candidates_mask, 0.0).float()  # tmp_batch x num_node
            graph_mask_squared = torch.bmm(graph_mask.unsqueeze(-1), graph_mask.unsqueeze(1))  # batch x num_node x num_node
            graph_self_attended, _ = self.online_net.sp_self_attention(graph_rep, graph_mask_squared, graph_rep, graph_rep)
            avg_graph_self_attended = masked_mean(graph_self_attended, graph_mask, dim=1)  # tmp_batch x hid*3

            # unsqueeze avg_attended_h_g_cand and avg_attended_cand_h_g
            for i, which in enumerate(tmp_batch_from):
                list_avg_graph_representation[which].append(avg_graph_self_attended[i])
        # paddings
        tensor_avg_graph_representation = torch.autograd.Variable(torch.zeros(batch_size, num_candidates, list_avg_graph_representation[0][0].size(-1)))  # batch x num_cand x hid*3
        if self.use_cuda:
            tensor_avg_graph_representation = tensor_avg_graph_representation.cuda()
        for b in range(batch_size):
            for c in range(len(list_avg_graph_representation[b])):
                tensor_avg_graph_representation[b, c, :] = list_avg_graph_representation[b][c]

        sp_ret = tensor_avg_graph_representation  # batch x num_cand x hid*3
        sp_ret = self.online_net.sp_linear_1(sp_ret)  # batch x num_cand x hid
        sp_ret = torch.relu(sp_ret)
        sp_ret = sp_ret * cand_mask.unsqueeze(-1)
        sp_ret = self.online_net.sp_linear_2(sp_ret).squeeze(-1)  # batch x num_cand
        sp_ret = sp_ret * cand_mask

        # masked softmax and NLL loss with mask
        sp_ret = sp_ret.masked_fill((1.0 - cand_mask).bool(), float('-inf'))
        loss = torch.nn.CrossEntropyLoss()(sp_ret, torch.argmax(pt_labels, -1))

        return loss, to_np(sp_ret), np_labels, graph_candidate_triplets_list

    ##################################
    # deep graph infomax specific
    ##################################

    def get_deep_graph_infomax_logits(self, triplets):
        batch_size = len(triplets)

        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        adjacency_matrices = self.get_graph_adjacency_matrix(triplets)

        node_embeddings = self.online_net.get_graph_node_representations(input_node_name)  # num_node x hid
        relation_embeddings = self.online_net.get_graph_relation_representations(input_relation_name)  # num_relation x hid
        node_embeddings = node_embeddings.repeat(batch_size, 1, 1)  # batch x num_node x hid
        relation_embeddings = relation_embeddings.repeat(batch_size, 1, 1)  # batch x num_relation x hid

        label_positive = torch.ones(batch_size, node_embeddings.size(1))
        label_negative = torch.zeros(batch_size, node_embeddings.size(1))
        labels = torch.cat((label_positive, label_negative), 1)  # batch x num_node*2
        if self.use_cuda:
            labels = labels.cuda()

        # get node mask
        node_mask = torch.sum(adjacency_matrices[:, :-1, :, :], 1)  # batch x num_nodes x num_nodes
        node_mask = torch.sum(node_mask, -1) + torch.sum(node_mask, -2)  # batch x num_node
        node_mask = torch.gt(node_mask, 0).float()  # batch x num_node

        shuffled_node_embeddings = node_embeddings.clone()
        batch_nonzero_idx = []
        for b in range(batch_size):
            nonzero_idx = node_mask[b].nonzero().squeeze(1)
            nonzero_idx_np = to_np(nonzero_idx)
            shuffled_idx_np = np.random.permutation(nonzero_idx_np)
            shuffled_idx = to_pt(shuffled_idx_np, self.use_cuda, type='long')
            shuffled_node_embeddings[b][nonzero_idx] = node_embeddings[b][shuffled_idx]
            batch_nonzero_idx.append(nonzero_idx_np)

        h_positive, h_negative, global_representations = self.online_net.get_deep_graph_infomax_discriminator_input(node_embeddings, shuffled_node_embeddings, node_mask, relation_embeddings, adjacency_matrices)
        dgi_discriminator_logits = self.online_net.dgi_discriminator(global_representations, h_positive, h_negative, self.sample_bias_positive, self.sample_bias_negative)  # batch x num_node*2

        loss = torch.nn.BCEWithLogitsLoss(reduction="none")(dgi_discriminator_logits, labels)
        loss = torch.sum(loss * torch.cat((node_mask, node_mask), 1), 1) / (2 * node_mask.sum(1))
        loss = torch.mean(loss)

        return loss, to_np(labels), to_np(dgi_discriminator_logits), batch_nonzero_idx

    ##################################
    # RL specific
    ##################################

    def finish_of_episode(self, episode_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.epsilon = max(self.epsilon, 0.0)

    def get_game_info_at_certain_step_fully_observable(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        # get new facts
        current_triplets = []  # batch of list of triplets
        for b in range(batch_size):
            new_f = set(process_fully_obs_facts(infos["game"][b], infos["facts"][b]))
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)

        return observation_strings, current_triplets, action_candidate_list, None, None

    def get_game_info_at_certain_step(self, obs, infos, prev_actions=None, prev_facts=None, return_gt_commands=False):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)

        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        # get new facts
        new_facts = []
        current_triplets = []  # batch of list of triplets
        commands_from_env = []  # batch of list of commands
        for b in range(batch_size):
            if prev_facts is None:
                new_f = process_facts(None, infos["game"][b], infos["facts"][b], None, None)
                prev_f = set()
            else:
                new_f = process_facts(prev_facts[b], infos["game"][b], infos["facts"][b], infos["last_action"][b], prev_actions[b])
                prev_f = prev_facts[b]
            new_facts.append(new_f)
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)
            target_commands = gen_graph_commands(new_f - prev_f, cmd="add") + gen_graph_commands(prev_f - new_f, cmd="delete")
            commands_from_env.append(target_commands)

        target_command_strings = []
        if return_gt_commands:
            # sort target commands and add seperators.
            target_command_strings = [" <sep> ".join(sort_target_commands(tgt_cmds)) for tgt_cmds in commands_from_env]

        return observation_strings, current_triplets, action_candidate_list, target_command_strings, new_facts

    def get_game_info_at_certain_step_lite(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)

        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        return observation_strings, action_candidate_list

    def update_knowledge_graph_triplets(self, triplets, prediction_strings):
        new_triplets = []
        for i in range(len(triplets)):
            # per example in a batch
            predict_cmds = prediction_strings[i].split("<sep>")
            if predict_cmds[-1].endswith("<eos>"):
                predict_cmds[-1] = predict_cmds[-1][:-5].strip()
            else:
                predict_cmds = predict_cmds[:-1]
            if len(predict_cmds) == 0:
                new_triplets.append(triplets[i])
                continue
            predict_cmds = [" ".join(item.split()) for item in predict_cmds]
            predict_cmds = [item for item in predict_cmds if len(item) > 0]
            new_triplets.append(update_graph_triplets(triplets[i], predict_cmds, self.node_vocab, self.relation_vocab))
        return new_triplets

    def encode(self, observation_strings, graph_input, use_model):
        assert self.task == "rl"
        model = self.choose_model(use_model)
        # step 1 and 3, at step 3, the agent doesn't have to re-encode observation
        # because it's essentially the same as in step 1
        if self.enable_text_input:
            obs_encoding_sequence, obs_mask = self.encode_text(observation_strings, use_model=use_model)
        else:
            obs_encoding_sequence, obs_mask = None, None

        if self.enable_graph_input:
            node_encoding_sequence, node_mask = self.encode_graph(graph_input, use_model=use_model)
        else:
            node_encoding_sequence, node_mask = None, None

        if self.enable_text_input and self.enable_graph_input:
            h_og, h_go = model.get_match_representations(obs_encoding_sequence, obs_mask, node_encoding_sequence, node_mask)
            return h_og, obs_mask, h_go, node_mask
        else:
            return obs_encoding_sequence, obs_mask, node_encoding_sequence, node_mask

    def action_scoring(self, action_candidate_list, h_og=None, obs_mask=None, h_go=None, node_mask=None, previous_h=None, previous_c=None, use_model=None):
        model = self.choose_model(use_model)
        # step 4
        input_action_candidate = self.get_action_candidate_list_input(action_candidate_list)
        action_scores, action_masks, new_h, new_c = model.score_actions(input_action_candidate, h_og, obs_mask, h_go, node_mask, previous_h, previous_c)  # batch x num_actions
        return action_scores, action_masks, new_h, new_c

    # action scoring stuff (Deep Q-Learning)
    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_maxQ_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    def act_greedy(self, observation_strings, graph_input, action_candidate_list, previous_h=None, previous_c=None):
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask = self.encode(observation_strings, graph_input, use_model="online")
            action_scores, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, new_h, new_c

    def act_random(self, observation_strings, graph_input, action_candidate_list, previous_h=None, previous_c=None):
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask = self.encode(observation_strings, graph_input, use_model="online")
            action_scores, _, new_h, new_c = self.action_scoring(action_candidate_list, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, new_h, new_c

    def act(self, observation_strings, graph_input, action_candidate_list, previous_h=None, previous_c=None, random=False):

        with torch.no_grad():
            if self.mode == "eval":
                return self.act_greedy(observation_strings, graph_input, action_candidate_list, previous_h, previous_c)
            if random:
                return self.act_random(observation_strings, graph_input, action_candidate_list, previous_h, previous_c)
            batch_size = len(observation_strings)

            h_og, obs_mask, h_go, node_mask = self.encode(observation_strings, graph_input, use_model="online")
            action_scores, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="online")

            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, new_h, new_c

    def get_dqn_loss_with_real_graphs(self, episode_no):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.sample(self.replay_batch_size, beta=self.beta_scheduler.value(episode_no), multi_step=self.multi_step)
        if data is None:
            return None, None

        obs_list, prev_action_list, candidate_list, action_indices, prev_graph_hidden_state, rewards, next_obs_list, next_prev_action_list, next_candidate_list, next_prev_graph_hidden_state, actual_indices, actual_ns, prior_weights = data
        prev_graph_hidden_state = to_pt(np.stack(prev_graph_hidden_state, 0), enable_cuda=self.use_cuda, type='float')
        new_adjacency_matrix, _ = self.generate_adjacency_matrix_for_rl(obs_list, prev_action_list, prev_graph_hidden_state)
        next_prev_graph_hidden_state = to_pt(np.stack(next_prev_graph_hidden_state, 0), enable_cuda=self.use_cuda, type='float')
        next_new_adjacency_matrix, _ = self.generate_adjacency_matrix_for_rl(next_obs_list, next_prev_action_list, next_prev_graph_hidden_state)

        h_og, obs_mask, h_go, node_mask = self.encode(obs_list, new_adjacency_matrix, use_model="online")
        action_scores, _, _, _ = self.action_scoring(candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_new_adjacency_matrix, use_model="online")
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_new_adjacency_matrix, use_model="target")
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards, reduce=False)  # batch

        prior_weights = to_pt(prior_weights, enable_cuda=self.use_cuda, type="float")
        loss = loss * prior_weights
        loss = torch.mean(loss)

        abs_td_error = np.abs(to_np(q_value - rewards))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        success = self.dqn_memory.update_priorities(actual_indices, new_priorities)
        if not success:
            return None, None

        return loss, q_value

    def get_dqn_loss(self, episode_no):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None

        data = self.dqn_memory.sample(self.replay_batch_size, beta=self.beta_scheduler.value(episode_no), multi_step=self.multi_step)

        if data is None:
            return None, None

        obs_list, _, candidate_list, action_indices, graph_triplet_list, rewards, next_obs_list, _, next_candidate_list, next_graph_triplet_list, actual_indices, actual_ns, prior_weights = data

        h_og, obs_mask, h_go, node_mask = self.encode(obs_list, graph_triplet_list, use_model="online")
        action_scores, _, _, _ = self.action_scoring(candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_graph_triplet_list, use_model="online")
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_graph_triplet_list, use_model="target")
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards, reduce=False)  # batch

        prior_weights = to_pt(prior_weights, enable_cuda=self.use_cuda, type="float")
        loss = loss * prior_weights
        loss = torch.mean(loss)

        abs_td_error = np.abs(to_np(q_value - rewards))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        self.dqn_memory.update_priorities(actual_indices, new_priorities)

        return loss, q_value

    def get_drqn_loss(self, episode_no):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None

        data = self.dqn_memory.sample_sequence(self.replay_batch_size, beta=self.beta_scheduler.value(episode_no), sample_history_length=self.replay_sample_history_length)

        if data is None:
            return None, None

        # obs_list, candidate_list, action_indices, graph_triplet_list, rewards, next_obs_list, next_candidate_list, next_graph_triplet_list
        actual_indices, prior_weights = data[-2], data[-1]
        loss_list, td_error_list, q_value_list = [], [], []
        prev_h, prev_c = None, None

        for step_no in range(self.replay_sample_history_length):
            obs_list = data[step_no][0]
            candidate_list = data[step_no][1]
            action_indices = data[step_no][2]
            graph_triplet_list = data[step_no][3]
            rewards = data[step_no][4]
            next_obs_list = data[step_no][5]
            next_candidate_list = data[step_no][6]
            next_graph_triplet_list = data[step_no][7]

            h_og, obs_mask, h_go, node_mask = self.encode(obs_list, graph_triplet_list, use_model="online")
            action_scores, _, new_h, new_c = self.action_scoring(candidate_list, h_og, obs_mask, h_go, node_mask, prev_h, prev_c, use_model="online")

            # ps_a
            action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

            prev_h, prev_c = new_h, new_c
            if step_no < self.replay_sample_update_from:
                q_value = q_value.detach()
                prev_h, prev_c = prev_h.detach(), prev_c.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)
                h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_graph_triplet_list, use_model="online")
                next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, prev_h, prev_c, use_model="online")

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
                next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
                # pns # Probabilities p(s_t+n, ·; θtarget)
                h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_graph_triplet_list, use_model="target")
                next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, prev_h, prev_c, use_model="target")

                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch

            rewards = rewards + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, rewards, reduce=False)  # batch

            p_weights = to_pt(prior_weights[step_no], enable_cuda=self.use_cuda, type="float")
            loss = loss * p_weights
            loss_list.append(loss)

            abs_td_error = np.abs(to_np(q_value - rewards))
            td_error_list.append(abs_td_error)
            q_value_list.append(q_value)

        for i in range(self.replay_sample_history_length - self.replay_sample_update_from):
            td_error = td_error_list[i]
            new_priorities = td_error + self.prioritized_replay_eps
            self.dqn_memory.update_priorities([item + self.replay_sample_update_from + i for item in actual_indices], new_priorities)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()

        return loss, q_value

    def update_dqn(self, episode_no):
        # update neural model by replaying snapshots in replay memory
        if self.real_valued_graph:
            dqn_loss, q_value = self.get_dqn_loss_with_real_graphs(episode_no)
        elif self.enable_recurrent_memory:
            dqn_loss, q_value = self.get_drqn_loss(episode_no)
        else:
            dqn_loss, q_value = self.get_dqn_loss(episode_no)
        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    def get_graph_rewards(self, prev_triplets, current_triplets):
        batch_size = len(current_triplets)
        if self.graph_reward_lambda == 0:
            return [0.0 for _ in current_triplets]

        if self.graph_reward_type == "triplets_increased":
            rewards = [float(len(c_triplet) - len(p_triplet)) for p_triplet, c_triplet in zip(prev_triplets, current_triplets)]
        elif self.graph_reward_type == "triplets_difference":
            rewards = []
            for b in range(batch_size):
                curr = current_triplets[b]
                prev = prev_triplets[b]
                curr = set(["|".join(item) for item in curr])
                prev = set(["|".join(item) for item in prev])
                diff_num = len(prev - curr) + len(curr - prev)
                rewards.append(float(diff_num))
        else:
            raise NotImplementedError
        rewards = [min(1.0, max(0.0, float(item) * self.graph_reward_lambda)) for item in rewards]
        return rewards

    def reset_binarized_counter(self, batch_size):
        self.binarized_counter_dict = [{} for _ in range(batch_size)]

    def get_binarized_count(self, observation_strings, update=True):
        batch_size = len(observation_strings)
        count_rewards = []
        for i in range(batch_size):
            concat_string = observation_strings[i]
            if concat_string not in self.binarized_counter_dict[i]:
                self.binarized_counter_dict[i][concat_string] = 0.0
            if update:
                self.binarized_counter_dict[i][concat_string] += 1.0
            r = self.binarized_counter_dict[i][concat_string]
            r = float(r == 1.0)
            count_rewards.append(r)
        return count_rewards
