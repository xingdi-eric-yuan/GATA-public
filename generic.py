import torch
import random
import uuid
import argparse
import copy
import os
import re
import yaml
import time
import string
import numpy as np
from collections import Counter
from os.path import join as pjoin
from functools import lru_cache

import textworld
from textworld.logic import State, Rule, Proposition, Variable
missing_words = set()


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor))


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        ids.append(_word_to_id(word, word2id))
    return ids


def _word_to_id(word, word2id):
    try:
        return word2id[word]
    except KeyError:
        key = word + "_" + str(len(word2id))
        if key not in missing_words:
            print("Warning... %s is not in vocab, vocab size is %d..." % (word, len(word2id)))
            missing_words.add(key)
            with open("missing_words.txt", 'a+') as outfile:
                outfile.write(key + '\n')
                outfile.flush()
        return 1


def max_len(list_of_list):
    if len(list_of_list) == 0:
        return 0
    return max(map(len, list_of_list))


def max_tensor_len(list_of_tensor, dim):
    tmp = []
    for t in list_of_tensor:
        tmp.append(t.size(dim))
    return max(tmp)


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    if isinstance(sequences, np.ndarray):
        return sequences
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_special_tokens(text):
        return re.sub(r'(<bos>|<eos>|<sep>|<pad>|<unk>)', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_special_tokens(lower(s)))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1.0
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def precision_recall_f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1.0, 1.0, 1.0
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0., 0., 0.
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def f1_score_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = f1_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i][0]])
    return torch.stack(res, 0)


def list_of_token_list_to_char_input(list_of_token_list, char2id):
    batch_size = len(list_of_token_list)
    max_token_number = max_len(list_of_token_list)
    max_char_number = max([max_len(item) for item in list_of_token_list])
    if max_char_number < 6:
        max_char_number = 6
    res = np.zeros((batch_size, max_token_number, max_char_number), dtype='int32')
    for i in range(batch_size):
        for j in range(len(list_of_token_list[i])):
            for k in range(len(list_of_token_list[i][j])):
                res[i][j][k] = _word_to_id(list_of_token_list[i][j][k], char2id)
    return res


def list_of_word_id_list_to_char_input(list_of_word_id_list, id2word, char2id):
    res = []
    for i in range(len(list_of_word_id_list)):
        res.append([id2word[item] for item in list_of_word_id_list[i]])
    return list_of_token_list_to_char_input(res, char2id)


def get_match_result_obs_gen(prediction_string, groundtruth_string):
    pred_string = prediction_string.split("<eos>")[0].rstrip()
    gt_string = groundtruth_string.split("<eos>")[0].rstrip()
    return precision_recall_f1_score(pred_string, gt_string)


def get_match_result(prediction_string, groundtruth_string, type='exact'):
    predict_cmds = prediction_string.split("<sep>")
    if predict_cmds[-1].endswith("<eos>"):
        predict_cmds[-1] = predict_cmds[-1][:-5].strip()
    else:
        predict_cmds = predict_cmds[:-1]

    groundtruth_cmds = groundtruth_string.split("<sep>")

    predict_cmds = [" ".join(item.split()) for item in predict_cmds]
    groundtruth_cmds = [" ".join(item.split()) for item in groundtruth_cmds]
    predict_cmds = [item for item in predict_cmds if len(item) > 0]
    groundtruth_cmds = [item for item in groundtruth_cmds if len(item) > 0]

    if len(predict_cmds) == 0 and len(groundtruth_cmds) == 0:
        return 1.0, 1.0, 1.0
    elif len(predict_cmds) == 0 or len(groundtruth_cmds) == 0:
        return 0.0, 0.0, 0.0

    predict_cmds = list(set(predict_cmds))
    groundtruth_cmds = list(set(groundtruth_cmds))

    match_score = np.asarray([0.0] * len(predict_cmds), dtype='float32')

    for pred_id, pred in enumerate(predict_cmds):
        if type == 'exact':
            if pred in groundtruth_cmds:
                match_score[pred_id] = 1
        elif type == 'soft':
            match_score[pred_id] = f1_score_over_ground_truths(pred, groundtruth_cmds)

    precision = float(sum(match_score)) / float(len(predict_cmds)) if len(predict_cmds) > 0 else 0.0
    recall = float(sum(match_score)) / float(len(groundtruth_cmds)) if len(groundtruth_cmds) > 0 else 0.0

    if precision + recall > 0:
        f1 = float(2 * (precision * recall)) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def power(a, b):
    if a == 0 or b == 0.0:
        return 1.0
    elif b > 0.0:
        return a ** b
    else:
        return 1.0 / (a ** -b)


def exists_triplet(triplets, arg1, arg2, relation):
    for i, t in enumerate(triplets):
        if arg1 in [t[0], "*"] and\
           arg2 in [t[1], "*"] and\
           relation in [t[2], "*"]:
           return i
        elif arg2 in [t[0], "*"] and\
             arg1 in [t[1], "*"] and\
             relation in [t[2], "*"]:
           return i
    return None


def update_graph_triplets(triplets, commands, node_vocab, relation_vocab):
    # remove duplicate but remain the order
    tmp_commands = []
    for cmd in commands:
        if cmd not in tmp_commands:
            tmp_commands.append(cmd)
    commands = tmp_commands
    for cmd in commands:
        # get verb-arg1-arg2
        if not (cmd.startswith("add") or cmd.startswith("delete")):
            continue
        cmd = cmd.split()
        if len(cmd) <= 3:
            continue
        verb = cmd[0]
        relation = cmd[-1]
        if relation not in relation_vocab:
            continue
        nouns = " ".join(cmd[1:-1])
        arg1, arg2 = "", ""
        for n in node_vocab:
            if nouns.startswith(n):
                tmp = nouns[len(n):].strip()
                if tmp == n:
                    continue
                if tmp in node_vocab:
                    arg1 = n
                    arg2 = tmp
                    break
        if arg1 == "" or arg2 == "":
            continue
        # manipulate KG
        index = exists_triplet(triplets, arg1, arg2, relation)
        if verb == "add":
            if index is not None:
                continue
            triplets.append([arg1, arg2, relation])
        else:
            if index is None:
                continue
            triplets = triplets[:index] + triplets[index + 1:]
    return triplets


def preproc(s, tokenizer=None):
    if s is None:
        return "nothing"
    s = s.replace("\n", ' ')
    if "$$$$$$$" in s:
        s = s.split("$$$$$$$")[-1]
    while(True):
        if "  " in s:
            s = s.replace("  ", " ")
        else:
            break
    s = s.strip()
    if len(s) == 0:
        return "nothing"
    s = " ".join([t.text for t in tokenizer(s)])
    s = s.lower()
    return s


##############################
# KG stuff
##############################
# relations
two_args_relations = ["in", "on", "at", "west_of", "east_of", "north_of", "south_of", "part_of", "needs"]
one_arg_state_relations = ["chopped", "roasted", "diced", "burned", "open", "fried", "grilled", "consumed", "closed", "sliced", "uncut", "raw"]
ignore_relations = ["cuttable", "edible", "drinkable", "sharp", "inedible", "cut", "cooked", "cookable", "needs_cooking"]
opposite_relations = {"west_of": "east_of",
                      "east_of": "west_of",
                      "south_of": "north_of",
                      "north_of": "south_of"}
equivalent_entities = {"inventory": "player",
                       "recipe": "cookbook"}
FOOD_FACTS = ["sliced", "diced", "chopped", "cut", "uncut", "cooked", "burned",
              "grilled", "fried", "roasted", "raw", "edible", "inedible"]


def process_equivalent_entities_in_triplet(triplet):
    # ["cookbook", "inventory", "in"]
    for i in range(len(triplet)):
        if triplet[i] in equivalent_entities:
            triplet[i] = equivalent_entities[triplet[i]]
    return triplet


def process_equivalent_entities_in_command(command):
    # "add , knife , inventory , in"
    words = command.split(" , ")
    words = [item.strip() for item in words]
    for i in range(len(words)):
        if words[i] in equivalent_entities:
            words[i] = equivalent_entities[words[i]]
    return " , ".join(words)


def process_exits_in_triplet(triplet):
    # ["exit", "kitchen", "backyard", "south_of"]
    if triplet[0] == "exit":
        return [triplet[0], triplet[1], triplet[3]]
    else:
        return triplet


def process_burning_triplets(list_of_triplets):
    burned_stuff = []
    for t in list_of_triplets:
        if "burned" in t:
            burned_stuff.append(t[0])
    res = []
    for t in list_of_triplets:
        if t[0] in burned_stuff and t[1] in ["grilled", "fried", "roasted"]:
            continue
        res.append(t)
    return res


def process_burning_commands(list_of_commands, list_of_triplets):
    cook = set(["grilled", "fried", "roasted"])
    burned_stuff = []
    for c in list_of_commands:
        if "burned" in c:
            burned_stuff.append(c.split(",")[1].strip())
    res = []
    for bs in burned_stuff:
        for t in list_of_triplets:
            if bs not in t:
                continue
            intersection = set(t) & cook
            if len(intersection) == 0:
                continue
            res.append("delete , " + bs + " , " + list(intersection)[0] + " , is")
            break
    return list_of_commands +  res


def process_direction_triplets(list_of_triplets):
    res = []
    for t in list_of_triplets:
        res.append(t)
        if t[0] == "exit" or t[1] == "exit":
            continue
        if "north_of" in t:
            res.append([t[1], t[0], "south_of"])
        elif "south_of" in t:
            res.append([t[1], t[0], "north_of"])
        elif "east_of" in t:
            res.append([t[1], t[0], "west_of"])
        elif "west_of" in t:
            res.append([t[1], t[0], "east_of"])
    return res


def process_direction_commands(list_of_commands):
    res = list(set(list_of_commands))
    return res


def arguments_swap(cmd, previous_action=None):
    if previous_action is None:
        return cmd
    previous_action = previous_action.lower()
    items = cmd.split(" , ")
    if items[-1] == "west_of" and previous_action == "go east":
        tmp = items[1]
        items[1] = items[2]
        items[2] = tmp
        items[-1] = "east_of"
    elif items[-1] == "north_of" and previous_action == "go south":
        tmp = items[1]
        items[1] = items[2]
        items[2] = tmp
        items[-1] = "south_of"
    return " , ".join(items)


def sort_target_commands(list_of_cmds):
    list_of_cmds = [item.split(" , ") for item in list_of_cmds]

    list_of_cmds.sort(key=lambda i: (i[0] == "add",  # add always before delete
                                        i[1] == "player",  # relations with player always first
                                        i[2] == "player",  # relations with player always first
                                        i[3] in ["west_of", "east_of", "north_of", "south_of"],  # room connections always first
                                        i[3] in ["part_of"],  # recipe
                                        i[3] in two_args_relations,  # two args relations first
                                        i[3] in ["is"],  # one arg state relations first
                                        i[3] in ["needs"],  # one arg requirement relations first
                                        i[2],
                                        i[1]))
    list_of_cmds = [" ".join(item) for item in list_of_cmds]
    list_of_cmds = list_of_cmds[::-1]
    res = []
    for cmd in list_of_cmds:
        if cmd not in res:
            res.append(cmd)
    return res


@lru_cache()
def _rules_predicates_scope():
    rules = [
        Rule.parse("query :: at(P, r) -> at(P, r)"),
        Rule.parse("query :: at(P, r) & at(o, r) -> at(o, r)"),
        Rule.parse("query :: at(P, r) & at(d, r) -> at(d, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) -> at(s, r)"),
        Rule.parse("query :: at(P, r) & at(c, r) -> at(c, r)"),
        Rule.parse("query :: at(P, r) & at(s, r) & on(o, s) -> on(o, s)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) -> open(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & closed(c) -> closed(c)"),
        Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(o, c) -> in(o, c)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & open(d) -> open(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & closed(d) -> closed(d)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: at(P, r) & link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    rules += [Rule.parse("query :: at(P, r) & at(f, r) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(s, r) & on(f, s) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    rules += [Rule.parse("query :: at(P, r) & at(c, r) & open(c) & in(f, c) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_predicates_recipe():
    rules = [
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) -> part_of(f, RECIPE)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & roasted(ingredient) -> needs_roasted(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & grilled(ingredient) -> needs_grilled(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & fried(ingredient) -> needs_fried(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & sliced(ingredient) -> needs_sliced(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & chopped(ingredient) -> needs_chopped(f)"),
        Rule.parse("query :: in(ingredient, RECIPE) & base(f, ingredient) & diced(ingredient) -> needs_diced(f)"),
    ]
    return rules


@lru_cache()
def _rules_exits():
    rules = [
        Rule.parse("query :: at(P, r) & north_of(r', r) -> north_of(r', r)"),
        Rule.parse("query :: at(P, r) & west_of(r', r) -> west_of(r', r)"),
        Rule.parse("query :: at(P, r) & south_of(r', r) -> south_of(r', r)"),
        Rule.parse("query :: at(P, r) & east_of(r', r) -> east_of(r', r)"),
    ]
    return rules


@lru_cache()
def _rules_predicates_inv():
    rules = [
        Rule.parse("query :: in(o, I) -> in(o, I)"),
    ]
    rules += [Rule.parse("query :: in(f, I) & {fact}(f) -> {fact}(f)".format(fact=fact)) for fact in FOOD_FACTS]
    return rules


@lru_cache()
def _rules_to_convert_link_predicates():
    rules = [
        Rule.parse("query :: link(r, d, r') & north_of(r', r) -> north_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & south_of(r', r) -> south_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & west_of(r', r) -> west_of(d, r)"),
        Rule.parse("query :: link(r, d, r') & east_of(r', r) -> east_of(d, r)"),
    ]
    return rules


def find_predicates_in_scope(state):
    actions = state.all_applicable_actions(_rules_predicates_scope())
    return [action.postconditions[0] for action in actions]


def find_exits_in_scope(state):
    actions = state.all_applicable_actions(_rules_exits())

    def _convert_to_exit_fact(proposition):
        return Proposition(proposition.name,
                           [Variable("exit", "LOCATION"),
                            proposition.arguments[1],
                            proposition.arguments[0]])

    return [_convert_to_exit_fact(action.postconditions[0]) for action in actions]


def convert_link_predicates(state):
    actions = state.all_applicable_actions(_rules_to_convert_link_predicates())
    for action in list(actions):
        state.apply(action)
    return state


def find_predicates_in_inventory(state):
    actions = state.all_applicable_actions(_rules_predicates_inv())
    return [action.postconditions[0] for action in actions]


def find_predicates_in_recipe(state):
    actions = state.all_applicable_actions(_rules_predicates_recipe())

    def _convert_to_needs_relation(proposition):
        if not proposition.name.startswith("needs_"):
            return proposition

        return Proposition("needs",
                           [proposition.arguments[0],
                            Variable(proposition.name.split("needs_")[-1], "STATE")])

    return [_convert_to_needs_relation(action.postconditions[0]) for action in actions]


def process_facts(prev_facts, info_game, info_facts, info_last_action, cmd):
    kb = info_game.kb
    if prev_facts is None or cmd == "restart":
        facts = set()
    else:
        if cmd == "inventory":  # Bypassing TextWorld's action detection.
            facts = set(find_predicates_in_inventory(State(kb.logic, info_facts)))
            return prev_facts | facts

        elif info_last_action is None :
            return prev_facts  # Invalid action, nothing has changed.

        elif info_last_action.name == "examine" and "cookbook" in [v.name for v in info_last_action.variables]:
            facts = set(find_predicates_in_recipe(State(kb.logic, info_facts)))
            return prev_facts | facts

        state = State(kb.logic, prev_facts | set(info_last_action.preconditions))
        success = state.apply(info_last_action)
        assert success
        facts = set(state.facts)

    # Always add facts in sight.
    facts |= set(find_predicates_in_scope(State(kb.logic, info_facts)))
    facts |= set(find_exits_in_scope(State(kb.logic, info_facts)))

    return facts


def process_fully_obs_facts(info_game, facts):
    state = State(info_game.kb.logic, facts)
    state = convert_link_predicates(state)
    inventory_facts = set(find_predicates_in_inventory(state))
    recipe_facts = set(find_predicates_in_recipe(state))
    return set(state.facts) | inventory_facts | recipe_facts


def process_local_obs_facts(info_game, info_facts, info_last_action, cmd):
    def _get_state():
        return State(info_game.kb.logic, info_facts)

    if cmd == "inventory":  # Bypassing TextWorld's action detection.
        return set(find_predicates_in_inventory(_get_state()))

    elif (info_last_action and info_last_action.name.startswith("go")) or cmd in ["restart", "look"]:
        # Facts in sight.
        state = _get_state()
        facts = set(find_predicates_in_scope(state))
        facts |= set(find_exits_in_scope(state))
        return facts

    elif info_last_action is None:
        return set()  # Invalid action, no facts.

    elif info_last_action.name == "examine" and "cookbook" in [v.name for v in info_last_action.variables]:
        return set(find_predicates_in_recipe(_get_state()))

    return info_last_action.postconditions


def serialize_facts(facts):
    PREDICATES_TO_DISCARD = {"ingredient_1", "ingredient_2", "ingredient_3", "ingredient_4", "ingredient_5",
                             "out", "free", "used", "cooking_location", "link"}
    CONSTANT_NAMES = {"P": "player", "I": "player", "ingredient": None, "slot": None, "RECIPE": "cookbook"}
    # e.g. [("wooden door", "backyard", "in"), ...]
    serialized = [[arg.name if arg.name and arg.type not in CONSTANT_NAMES else CONSTANT_NAMES[arg.type] for arg in fact.arguments] + [fact.name]
                    for fact in sorted(facts) if fact.name not in PREDICATES_TO_DISCARD]
    return filter_triplets([fact for fact in serialized if None not in fact])


def filter_triplets(triplets):
    tp = []
    for item in triplets:
        # item = process_equivalent_entities_in_triplet(item)
        item = process_exits_in_triplet(item)
        if item[-1] in (two_args_relations + one_arg_state_relations):
            tp.append([it.lower() for it in item])
        else:
            if item[-1] not in ignore_relations:
                print("Warning..., %s not in known relations..." % (item[-1]))

    for i in range(len(tp)):
        if tp[i][-1] in one_arg_state_relations:
            tp[i].append("is")

    tp = process_burning_triplets(tp)
    # tp = process_direction_triplets(tp)
    return tp


def gen_graph_commands(facts, cmd):
    # e.g ["add , player , shed , in", ...]
    return [" , ".join([cmd] + fact) for fact in serialize_facts(facts)]


class HistoryScoreCache:

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


def graph_triplets_to_string(list_of_triples):
    list_of_triples = ["|".join(item) for item in list_of_triples]
    list_of_triples.sort()
    key = "<|>".join(list_of_triples)
    return key


class EpisodicCountingMemory:

    def __init__(self):
        self.reset()

    def push(self, stuff):
        """stuff is list of list of list of strings.
           e.g.: [[['player', 'shed', 'at'], ['workbench', 'shed', 'at']]]
        """
        assert len(stuff) > 0  # batch size should be greater than 0
        if len(self.memory) == 0:
            for _ in range(len(stuff)):
                self.memory.append(set())

        for b in range(len(stuff)):
            key = graph_triplets_to_string(stuff[b])
            self.memory[b].add(key)

    def has_not_seen(self, stuff):
        assert len(stuff) > 0  # batch size should be greater than 0
        res = []
        for b in range(len(stuff)):
            key = graph_triplets_to_string(stuff[b])
            res.append(key not in self.memory[b])
        return res

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


def two_lists_of_lists_are_identical(list_1, list_2):
    return list_1 == list_2


def generate_labels_for_sp(target_graph, graph_candidates, k_way_classification=5):
    if k_way_classification == -1:
        new_graph_candidates = copy.deepcopy(graph_candidates)
        for b in range(len(new_graph_candidates)):
            random.shuffle(new_graph_candidates[b])
        return [[(two_lists_of_lists_are_identical(z, x)) * 1 for z in y] for (x, y) in zip(target_graph, new_graph_candidates)], new_graph_candidates

    batch_size = len(graph_candidates)
    negative_samples = []
    for b in range(batch_size):
        tmp = [item for item in graph_candidates[b] if not two_lists_of_lists_are_identical(item, target_graph[b])]
        ns_indices = np.random.choice(len(tmp), k_way_classification - 1, replace=True).tolist()
        ns = []
        for idx in ns_indices:
            ns.append(tmp[idx])
        negative_samples.append(ns)
    indices = np.random.randint(low=0, high=k_way_classification, size=batch_size).tolist()
    labels = np.zeros((batch_size, k_way_classification), dtype="int32")
    new_graph_candidates = []
    for b, which in enumerate(indices):
        labels[b][which] = 1
        tmp = negative_samples[b][:which] + [target_graph[b]] + negative_samples[b][which:]
        new_graph_candidates.append(tmp)
    return labels, new_graph_candidates


def generate_labels_for_ap(action_list, action_candidate_list, k_way_classification=5):
    if k_way_classification == -1:
        new_action_candidates = copy.deepcopy(action_candidate_list)
        for b in range(len(new_action_candidates)):
            random.shuffle(new_action_candidates[b])
        return [[(z == x) * 1 for z in y] for (x, y) in zip(action_list, new_action_candidates)], new_action_candidates

    batch_size = len(action_candidate_list)
    negative_samples = []
    for b in range(batch_size):
        ns = [item for item in action_candidate_list[b] if item != action_list[b]]
        ns = np.random.choice(ns, k_way_classification - 1, replace=True).tolist()
        negative_samples.append(ns)
    indices = np.random.randint(low=0, high=k_way_classification, size=batch_size).tolist()
    labels = np.zeros((batch_size, k_way_classification), dtype="int32")
    new_action_candidates = []
    for b, which in enumerate(indices):
        labels[b][which] = 1
        tmp = negative_samples[b][:which] + [action_list[b]] + negative_samples[b][which:]
        new_action_candidates.append(tmp)
    return labels, new_action_candidates


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.schedule = np.linspace(initial_p, final_p, schedule_timesteps)

    def value(self, step):
        if step < 0:
            return self.initial_p
        if step >= self.schedule_timesteps:
            return self.final_p
        else:
            return self.schedule[step]


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)
    # print(config)
    return config
