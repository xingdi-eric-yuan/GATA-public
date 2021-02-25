import os
import copy
import logging
import numpy as np

import torch
from layers import Embedding, EncoderBlock, DecoderBlock, DecoderBlockForObsGen, CQAttention, StackedRelationalGraphConvolution
from layers import PointerSoftmax, masked_softmax, NoisyLinear, SelfAttention, LSTMCell, DGIDiscriminator, masked_mean, ObservationDiscriminator
from generic import to_pt

logger = logging.getLogger(__name__)


class KG_Manipulation(torch.nn.Module):
    model_name = 'kg_manipulation'

    def __init__(self, config, word_vocab, node_vocab, relation_vocab):
        super(KG_Manipulation, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(word_vocab)
        self.node_vocab = node_vocab
        self.node_vocab_size = len(node_vocab)
        self.relation_vocab = relation_vocab
        self.relation_vocab_size = len(relation_vocab)
        self.read_config()
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        model_config = self.config['general']['model']

        self.use_pretrained_embedding = model_config['use_pretrained_embedding']
        self.word_embedding_size = model_config['word_embedding_size']
        self.word_embedding_trainable = model_config['word_embedding_trainable']
        self.pretrained_embedding_path = "crawl-300d-2M.vec.h5"

        self.node_embedding_size = model_config['node_embedding_size']
        self.node_embedding_trainable = model_config['node_embedding_trainable']
        self.relation_embedding_size = model_config['relation_embedding_size']
        self.relation_embedding_trainable = model_config['relation_embedding_trainable']
        self.embedding_dropout = model_config['embedding_dropout']

        self.gcn_hidden_dims = model_config['gcn_hidden_dims']
        self.gcn_highway_connections = model_config['gcn_highway_connections']
        self.gcn_num_bases = model_config['gcn_num_bases']
        self.real_valued_graph = model_config['real_valued_graph']

        self.encoder_layers = model_config['encoder_layers']
        self.decoder_layers = model_config['decoder_layers']
        self.action_scorer_layers = model_config['action_scorer_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.attention_dropout = model_config['attention_dropout']
        self.block_dropout = model_config['block_dropout']
        self.dropout = model_config['dropout']

        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']

        self.enable_recurrent_memory = self.config['rl']['model']['enable_recurrent_memory']
        self.enable_graph_input = self.config['rl']['model']['enable_graph_input']
        self.enable_text_input = self.config['rl']['model']['enable_text_input']

    def _def_layers(self):

        # word embeddings
        if self.use_pretrained_embedding:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            id2word=self.word_vocab,
                                            dropout_rate=self.embedding_dropout,
                                            load_pretrained=True,
                                            trainable=self.word_embedding_trainable,
                                            embedding_oov_init="random",
                                            pretrained_embedding_path=self.pretrained_embedding_path)
        else:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            trainable=self.word_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        # node embeddings
        self.node_embedding = Embedding(embedding_size=self.node_embedding_size,
                                        vocab_size=self.node_vocab_size,
                                        trainable=self.node_embedding_trainable,
                                        dropout_rate=self.embedding_dropout)

        # relation embeddings
        self.relation_embedding = Embedding(embedding_size=self.relation_embedding_size,
                                            vocab_size=self.relation_vocab_size,
                                            trainable=self.relation_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        self.word_embedding_prj = torch.nn.Linear(self.word_embedding_size, self.block_hidden_dim, bias=False)
        self.encoder =  torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.rgcns = StackedRelationalGraphConvolution(entity_input_dim=self.node_embedding_size+self.block_hidden_dim, relation_input_dim=self.relation_embedding_size+self.block_hidden_dim, num_relations=self.relation_vocab_size, hidden_dims=self.gcn_hidden_dims, num_bases=self.gcn_num_bases,
        use_highway_connections=self.gcn_highway_connections, dropout_rate=self.dropout, real_valued_graph=self.real_valued_graph)
        self.attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)

        self.self_attention_text = SelfAttention(self.block_hidden_dim, self.n_heads, self.dropout)
        self.self_attention_graph = SelfAttention(self.block_hidden_dim, self.n_heads, self.dropout)

        # recurrent memories
        self.recurrent_memory_bi_input = LSTMCell(self.block_hidden_dim * 2, self.block_hidden_dim, use_bias=True)
        self.recurrent_memory_single_input = LSTMCell(self.block_hidden_dim, self.block_hidden_dim, use_bias=True)

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_linear_1_tri_input = linear_function(self.block_hidden_dim * 3, self.block_hidden_dim)
        self.action_scorer_linear_1_bi_input = linear_function(self.block_hidden_dim * 2, self.block_hidden_dim)
        self.action_scorer_linear_2 = linear_function(self.block_hidden_dim, 1)

        # text encoder for pretraining tasks
        # (we separate this because we don't want to init text encoder with pretrained parameters when training RL)
        self.encoder_for_pretraining_tasks =  torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        # command generation
        self.cmd_gen_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.cmd_gen_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.decoder = torch.nn.ModuleList([DecoderBlock(ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.decoder_layers)])
        self.tgt_word_prj = torch.nn.Linear(self.block_hidden_dim, self.word_vocab_size, bias=False)
        self.pointer_softmax = PointerSoftmax(input_dim=self.block_hidden_dim, hidden_dim=self.block_hidden_dim)

        # observation generation
        self.obs_gen_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.obs_gen_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.obs_gen_decoder = torch.nn.ModuleList([DecoderBlockForObsGen(ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.decoder_layers)])
        self.obs_gen_tgt_word_prj = torch.nn.Linear(self.block_hidden_dim, self.word_vocab_size, bias=False)
        self.obs_gen_linear_1 = torch.nn.Linear(self.block_hidden_dim, self.block_hidden_dim)
        self.obs_gen_linear_2 = torch.nn.Linear(self.block_hidden_dim, int(len(self.relation_vocab) / 2) * len(self.node_vocab) * len(self.node_vocab))
        self.obs_gen_attention_to_rnn_input = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)
        self.obs_gen_graph_rnncell = torch.nn.GRUCell(self.block_hidden_dim, self.block_hidden_dim)
        self.observation_discriminator = ObservationDiscriminator(self.block_hidden_dim)

        # action prediction
        self.ap_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.ap_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.ap_self_attention = SelfAttention(self.block_hidden_dim * 3, self.n_heads, self.dropout)
        self.ap_linear_1 = torch.nn.Linear(self.block_hidden_dim * 3, self.block_hidden_dim)
        self.ap_linear_2 = torch.nn.Linear(self.block_hidden_dim, 1)

        # state prediction
        self.sp_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.attention_dropout)
        self.sp_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.sp_self_attention = SelfAttention(self.block_hidden_dim * 3, self.n_heads, self.dropout)
        self.sp_linear_1 = torch.nn.Linear(self.block_hidden_dim * 3, self.block_hidden_dim)
        self.sp_linear_2 = torch.nn.Linear(self.block_hidden_dim, 1)

        # deep graph infomax
        self.dgi_discriminator = DGIDiscriminator(self.gcn_hidden_dims[-1])

    def embed(self, input_words):
        word_embeddings, mask = self.word_embedding(input_words)  # batch x time x emb
        word_embeddings = self.word_embedding_prj(word_embeddings)
        word_embeddings = word_embeddings * mask.unsqueeze(-1)  # batch x time x hid
        return word_embeddings, mask

    def encode_text(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder[i](encoding_sequence, squared_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def encode_text_for_pretraining_tasks(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder_for_pretraining_tasks[i](encoding_sequence, squared_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def get_graph_node_representations(self, node_names_word_ids):
        # node_names_word_ids: num_node x num_word
        node_name_embeddings, _mask = self.embed(node_names_word_ids)  # num_node x num_word x emb
        _mask = torch.sum(_mask, -1)  # num_node
        node_name_embeddings = torch.sum(node_name_embeddings, 1)  # num_node x hid
        tmp = torch.eq(_mask, 0).float()
        if node_name_embeddings.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        node_name_embeddings = node_name_embeddings / _mask.unsqueeze(-1)
        node_name_embeddings = node_name_embeddings.unsqueeze(0)  # 1 x num_node x emb

        node_ids = np.arange(self.node_vocab_size)  # num_node
        node_ids = to_pt(node_ids, enable_cuda=node_names_word_ids.is_cuda, type='long').unsqueeze(0)  # 1 x num_node
        node_embeddings, _ = self.node_embedding(node_ids)  # 1 x num_node x emb
        node_embeddings = torch.cat([node_name_embeddings, node_embeddings], dim=-1)  # 1 x num_node x emb+emb
        return node_embeddings

    def get_graph_relation_representations(self, relation_names_word_ids):
        # relation_names_word_ids: num_relation x num_word
        relation_name_embeddings, _mask = self.embed(relation_names_word_ids)  # num_relation x num_word x emb
        _mask = torch.sum(_mask, -1)  # num_relation
        relation_name_embeddings = torch.sum(relation_name_embeddings, 1)  # num_relation x hid
        tmp = torch.eq(_mask, 0).float()
        if relation_name_embeddings.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        relation_name_embeddings = relation_name_embeddings / _mask.unsqueeze(-1)
        relation_name_embeddings = relation_name_embeddings.unsqueeze(0)  # 1 x num_relation x emb

        relation_ids = np.arange(self.relation_vocab_size)  # num_relation
        relation_ids = to_pt(relation_ids, enable_cuda=relation_names_word_ids.is_cuda, type='long').unsqueeze(0)  # 1 x num_relation
        relation_embeddings, _ = self.relation_embedding(relation_ids)  # 1 x num_relation x emb
        relation_embeddings = torch.cat([relation_name_embeddings, relation_embeddings], dim=-1)  # 1 x num_relation x emb+emb
        return relation_embeddings

    def encode_graph(self, node_names_word_ids, relation_names_word_ids, input_adjacency_matrices):
        # node_names_word_ids: num_node x num_word
        # relation_names_word_ids: num_relation x num_word
        # input_adjacency_matrices: batch x num_relations x num_node x num_node
        # graph node embedding / encoding
        node_embeddings = self.get_graph_node_representations(node_names_word_ids)  # 1 x num_node x emb+emb
        relation_embeddings = self.get_graph_relation_representations(relation_names_word_ids)  # 1 x num_node x emb+emb
        node_embeddings = node_embeddings.repeat(input_adjacency_matrices.size(0), 1, 1)  # batch x num_node x emb+emb
        relation_embeddings = relation_embeddings.repeat(input_adjacency_matrices.size(0), 1, 1)  # batch x num_relation x emb+emb

        node_encoding_sequence = self.rgcns(node_embeddings, relation_embeddings, input_adjacency_matrices)  # batch x num_node x enc

        if self.real_valued_graph:
            node_mask = torch.ones(node_encoding_sequence.size(0), node_encoding_sequence.size(1))  # batch x num_node
            if node_encoding_sequence.is_cuda:
                node_mask.cuda()
        else:
            node_mask = torch.sum(input_adjacency_matrices[:, :-1, :, :], 1)  # batch x num_node x num_node
            node_mask = torch.sum(node_mask, -1) + torch.sum(node_mask, -2)  # batch x num_node
            node_mask = torch.gt(node_mask, 0).float()
            node_encoding_sequence = node_encoding_sequence * node_mask.unsqueeze(-1)

        return node_encoding_sequence, node_mask

    def get_match_representations(self, obs_encodings, obs_mask, node_encodings, node_mask):
        h_og = self.attention(obs_encodings, node_encodings, obs_mask, node_mask)
        h_go = self.attention(node_encodings, obs_encodings, node_mask, obs_mask)
        h_og = self.attention_prj(h_og)
        h_go = self.attention_prj(h_go)
        return h_og, h_go

    def get_subsequent_mask(self, seq):
        ''' For masking out the subsequent info. '''
        _, length = seq.size()
        subsequent_mask = torch.triu(torch.ones((length, length)), diagonal=1).float()
        subsequent_mask = 1.0 - subsequent_mask
        if seq.is_cuda:
            subsequent_mask = subsequent_mask.cuda()
        subsequent_mask = subsequent_mask.unsqueeze(0)  # 1 x time x time
        return subsequent_mask

    def decode_for_obs_gen(self, input_target_word_ids, h_ag2, prev_action_mask, h_ga2, node_mask):
        trg_embeddings, trg_mask = self.embed(input_target_word_ids) # batch x target_len x emb

        trg_mask_square = torch.bmm(trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1)) # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(input_target_word_ids) # batch x target_len x target_len

        prev_action_mask_square = torch.bmm(trg_mask.unsqueeze(-1), prev_action_mask.unsqueeze(1))
        node_mask_square = torch.bmm(trg_mask.unsqueeze(-1), node_mask.unsqueeze(1)) # batch x target_len x num_nodes

        trg_decoder_output = trg_embeddings
        for i in range(self.decoder_layers):
            trg_decoder_output, _ = self.obs_gen_decoder[i](trg_decoder_output, trg_mask, trg_mask_square, h_ag2, prev_action_mask_square, h_ga2, node_mask_square, i * 3 + 1, self.decoder_layers)

        trg_decoder_output = self.obs_gen_tgt_word_prj(trg_decoder_output)
        trg_decoder_output = masked_softmax(trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)
        # eliminating pointer softmax
        return trg_decoder_output
        
    def decode(self, input_target_word_ids, h_og, obs_mask, h_go, node_mask, input_obs):
        trg_embeddings, trg_mask = self.embed(input_target_word_ids)  # batch x target_len x emb

        trg_mask_square = torch.bmm(trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(input_target_word_ids)  # batch x target_len x target_len

        obs_mask_square = torch.bmm(trg_mask.unsqueeze(-1), obs_mask.unsqueeze(1))  # batch x target_len x obs_len
        node_mask_square = torch.bmm(trg_mask.unsqueeze(-1), node_mask.unsqueeze(1))  # batch x target_len x node_len

        trg_decoder_output = trg_embeddings
        for i in range(self.decoder_layers):
            trg_decoder_output, target_target_representations, target_source_representations, target_source_attention = self.decoder[i](trg_decoder_output, trg_mask, trg_mask_square, h_og, obs_mask_square, h_go, node_mask_square, i * 3 + 1, self.decoder_layers)  # batch x time x hid

        trg_decoder_output = self.tgt_word_prj(trg_decoder_output)
        trg_decoder_output = masked_softmax(trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)
        output = self.pointer_softmax(target_target_representations, target_source_representations, trg_decoder_output, trg_mask, target_source_attention, obs_mask, input_obs)

        return output

    def score_actions(self, input_candidate_word_ids, h_og=None, obs_mask=None, h_go=None, node_mask=None, previous_h=None, previous_c=None):
        # input_candidate_word_ids: batch x num_candidate x candidate_len
        batch_size, num_candidate, candidate_len = input_candidate_word_ids.size(0), input_candidate_word_ids.size(1), input_candidate_word_ids.size(2)
        input_candidate_word_ids = input_candidate_word_ids.view(batch_size * num_candidate, candidate_len)

        cand_encoding_sequence, cand_mask = self.encode_text(input_candidate_word_ids)
        cand_encoding_sequence = cand_encoding_sequence.view(batch_size, num_candidate, candidate_len, -1)
        cand_mask = cand_mask.view(batch_size, num_candidate, candidate_len)

        _mask = torch.sum(cand_mask, -1)  # batch x num_candidate
        candidate_representations = torch.sum(cand_encoding_sequence, -2)  # batch x num_candidate x hid
        tmp = torch.eq(_mask, 0).float()
        if candidate_representations.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        candidate_representations = candidate_representations / _mask.unsqueeze(-1)  # batch x num_candidate x hid
        cand_mask = cand_mask.byte().any(-1).float()  # batch x num_candidate

        if h_go is not None:
            node_mask_squared = torch.bmm(node_mask.unsqueeze(-1), node_mask.unsqueeze(1))  # batch x num_node x num_node
            graph_representations, _ = self.self_attention_graph(h_go, node_mask_squared, h_go, h_go)  # batch x num_node x hid
            # masked mean
            _mask = torch.sum(node_mask, -1)  # batch
            graph_representations = torch.sum(graph_representations, -2)  # batch x hid
            tmp = torch.eq(_mask, 0).float()
            if graph_representations.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            graph_representations = graph_representations / _mask.unsqueeze(-1)  # batch x hid

        if h_og is not None:
            obs_mask_squared = torch.bmm(obs_mask.unsqueeze(-1), obs_mask.unsqueeze(1))  # batch x obs_len x obs_len
            obs_representations, _ = self.self_attention_text(h_og, obs_mask_squared, h_og, h_og)  # batch x obs_len x hid
            # masked mean
            _mask = torch.sum(obs_mask, -1)  # batch
            obs_representations = torch.sum(obs_representations, -2)  # batch x hid
            tmp = torch.eq(_mask, 0).float()
            if obs_representations.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            obs_representations = obs_representations / _mask.unsqueeze(-1)  # batch x hid

        assert (h_og is not None) or (h_go is not None)
        if h_og is None:
            # only graph
            if self.enable_recurrent_memory:
                # recurrent memory
                new_h, new_c = self.recurrent_memory_single_input(graph_representations, h_0=previous_h, c_0=previous_c)
                new_h_expanded = torch.stack([new_h] * num_candidate, 1).view(batch_size, num_candidate, new_h.size(-1))
            else:
                new_h, new_c = None, None
                new_h_expanded = torch.stack([graph_representations] * num_candidate, 1).view(batch_size, num_candidate, graph_representations.size(-1))
            output = self.action_scorer_linear_1_bi_input(torch.cat([candidate_representations, new_h_expanded], -1))  # batch x num_candidate x hid
        elif h_go is None:
            # only text
            if self.enable_recurrent_memory:
                # recurrent memory
                new_h, new_c = self.recurrent_memory_single_input(obs_representations, h_0=previous_h, c_0=previous_c)
                new_h_expanded = torch.stack([new_h] * num_candidate, 1).view(batch_size, num_candidate, new_h.size(-1))
            else:
                new_h, new_c = None, None
                new_h_expanded = torch.stack([obs_representations] * num_candidate, 1).view(batch_size, num_candidate, obs_representations.size(-1))
            output = self.action_scorer_linear_1_bi_input(torch.cat([candidate_representations, new_h_expanded], -1))  # batch x num_candidate x hid
        else:
            # both available
            if self.enable_recurrent_memory:
                # recurrent memory
                new_h, new_c = self.recurrent_memory_bi_input(torch.cat([obs_representations, graph_representations], -1), h_0=previous_h, c_0=previous_c)
                new_h_expanded = torch.stack([new_h] * num_candidate, 1).view(batch_size, num_candidate, new_h.size(-1))
                output = self.action_scorer_linear_1_bi_input(torch.cat([candidate_representations, new_h_expanded], -1))  # batch x num_candidate x hid
            else:
                new_h, new_c = None, None
                obs_representations_expanded = torch.stack([obs_representations] * num_candidate, 1).view(batch_size, num_candidate, obs_representations.size(-1))
                graph_representations_expanded = torch.stack([graph_representations] * num_candidate, 1).view(batch_size, num_candidate, graph_representations.size(-1))
                output = self.action_scorer_linear_1_tri_input(torch.cat([candidate_representations, obs_representations_expanded, graph_representations_expanded], -1))  # batch x num_candidate x hid

        output = torch.relu(output)
        output = output * cand_mask.unsqueeze(-1)
        output = self.action_scorer_linear_2(output).squeeze(-1)  # batch x num_candidate
        output = output * cand_mask

        return output, cand_mask, new_h, new_c

    def get_deep_graph_infomax_discriminator_input(self, node_embeddings, shuffled_node_embeddings, node_masks, relation_embeddings, adjacency_matrix):
        
        h_positive = self.rgcns(node_embeddings, relation_embeddings, adjacency_matrix)
        h_positive = h_positive * node_masks.unsqueeze(-1)  # batch x num_node x hid

        h_negative = self.rgcns(shuffled_node_embeddings, relation_embeddings, adjacency_matrix)
        h_negative = h_negative * node_masks.unsqueeze(-1)  # batch x num_node x hid

        global_representations = masked_mean(h_positive, node_masks, dim=1)  # batch x hid
        global_representations = torch.sigmoid(global_representations)  # batch x hid
        return h_positive, h_negative, global_representations

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_linear_1_bi_input.reset_noise()
            self.action_scorer_linear_1_tri_input.reset_noise()
            self.action_scorer_linear_2.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.action_scorer_linear_1_bi_input.zero_noise()
            self.action_scorer_linear_1_tri_input.zero_noise()
            self.action_scorer_linear_2.zero_noise()
