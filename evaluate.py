import numpy as np
import torch
import os
from generic import get_match_result, to_np


def evaluate_with_ground_truth_graph(env, agent, num_games):
    # here we do not eval command generation
    achieved_game_points = []
    total_game_steps = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    while(True):
        if game_id >= num_games:
            break
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_name_list += [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list += [game.max_score for game in infos["game"]]

        batch_size = len(obs)
        agent.eval()
        agent.init()

        chosen_actions, prev_step_dones = [], []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_step_dones.append(0.0)
        prev_h, prev_c = None, None

        observation_strings, current_triplets, action_candidate_list, _, _ = agent.get_game_info_at_certain_step(obs, infos, prev_actions=None, prev_facts=None)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        still_running_mask = []
        final_scores = []

        for step_no in range(agent.eval_max_nb_steps_per_episode):

            # choose what to do next from candidate list
            chosen_actions, chosen_indices, _, prev_h, prev_c = agent.act_greedy(observation_strings, current_triplets, action_candidate_list, prev_h, prev_c)
            # send chosen actions to game engine
            chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_indices)]
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)

            observation_strings, current_triplets, action_candidate_list, _, _ = agent.get_game_info_at_certain_step(obs, infos, prev_actions=None, prev_facts=None)
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]

            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            final_scores = scores
            still_running_mask.append(still_running)

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        game_id += batch_size

    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normalized_game_points = achieved_game_points / game_max_score_list

    print_strings = []
    print_strings.append("======================================================")
    print_strings.append("EVAL: rewards: {:2.3f} | normalized reward: {:2.3f} | used steps: {:2.3f}".format(np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps)))
    for i in range(len(game_name_list)):
        print_strings.append("game name: {}, reward: {:2.3f}, normalized reward: {:2.3f}, steps: {:2.3f}".format(game_name_list[i], achieved_game_points[i], normalized_game_points[i], total_game_steps[i]))
    print_strings.append("======================================================")
    print_strings = "\n".join(print_strings)
    print(print_strings)
    return np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), 0.0, print_strings


def evaluate(env, agent, num_games):
    if agent.fully_observable_graph:
        return evaluate_with_ground_truth_graph(env, agent, num_games)

    achieved_game_points = []
    total_game_steps = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    while(True):
        if game_id >= num_games:
            break
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_name_list += [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list += [game.max_score for game in infos["game"]]
        batch_size = len(obs)
        agent.eval()
        agent.init()

        triplets, chosen_actions, prev_game_facts = [], [], []
        prev_step_dones = []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_game_facts.append(set())
            triplets.append([])
            prev_step_dones.append(0.0)
            
        prev_h, prev_c = None, None

        observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=None)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        still_running_mask = []
        final_scores = []

        for step_no in range(agent.eval_max_nb_steps_per_episode):

            # choose what to do next from candidate list
            chosen_actions, chosen_indices, _, prev_h, prev_c = agent.act_greedy(observation_strings, current_triplets, action_candidate_list, prev_h, prev_c)
            # send chosen actions to game engine
            chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_indices)]
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)

            prev_game_facts = current_game_facts
            observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=prev_game_facts)
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]

            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            final_scores = scores
            still_running_mask.append(still_running)

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        game_id += batch_size

    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normalized_game_points = achieved_game_points / game_max_score_list

    print_strings = []
    print_strings.append("======================================================")
    print_strings.append("EVAL: rewards: {:2.3f} | normalized reward: {:2.3f} | used steps: {:2.3f}".format(np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps)))
    for i in range(len(game_name_list)):
        print_strings.append("game name: {}, reward: {:2.3f}, normalized reward: {:2.3f}, steps: {:2.3f}".format(game_name_list[i], achieved_game_points[i], normalized_game_points[i], total_game_steps[i]))
    print_strings.append("======================================================")
    print_strings = "\n".join(print_strings)
    print(print_strings)
    return np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), 0.0, print_strings


def evaluate_belief_mode(env, agent, num_games):

    achieved_game_points = []
    total_game_steps = []
    total_command_generation_f1 = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    while(True):
        if game_id >= num_games:
            break
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_name_list += [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list += [game.max_score for game in infos["game"]]
        batch_size = len(obs)
        agent.eval()
        agent.init()

        triplets, chosen_actions, prev_game_facts = [], [], []
        avg_command_generation_f1_in_a_game, prev_step_dones = [], []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_game_facts.append(set())
            triplets.append([])
            avg_command_generation_f1_in_a_game.append([])
            prev_step_dones.append(0.0)
            
        prev_h, prev_c = None, None
        observation_strings, _, action_candidate_list, target_command_strings, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=None, return_gt_commands=True)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        still_running_mask = []
        final_scores = []

        for step_no in range(agent.eval_max_nb_steps_per_episode):

            # generate triplets to update the observed info into KG
            generated_commands = agent.command_generation_greedy_generation(observation_strings, triplets)
            triplets = agent.update_knowledge_graph_triplets(triplets, generated_commands)
            # choose what to do next from candidate list
            chosen_actions, chosen_indices, _, prev_h, prev_c = agent.act_greedy(observation_strings, triplets, action_candidate_list, prev_h, prev_c)
            # send chosen actions to game engine
            chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_indices)]
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)

            # eval command generation
            for i in range(batch_size):
                _, _, exact_f1 = get_match_result(generated_commands[i], target_command_strings[i], type='exact')
                avg_command_generation_f1_in_a_game[i].append(exact_f1)

            prev_game_facts = current_game_facts
            observation_strings, _, action_candidate_list, target_command_strings, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=prev_game_facts, return_gt_commands=True)
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]

            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            final_scores = scores
            still_running_mask.append(still_running)

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        total_command_generation_f1 += np.mean(avg_command_generation_f1_in_a_game, 1).tolist()
        game_id += batch_size

    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normalized_game_points = achieved_game_points / game_max_score_list

    print_strings = []
    print_strings.append("======================================================")
    print_strings.append("EVAL: rewards: {:2.3f} | normalized reward: {:2.3f} | used steps: {:2.3f} | command generation f1: {:2.3f}".format(np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), np.mean(total_command_generation_f1)))
    for i in range(len(game_name_list)):
        print_strings.append("game name: {}, reward: {:2.3f}, normalized reward: {:2.3f}, steps: {:2.3f}, cmd gen f1: {:2.3f}".format(game_name_list[i], achieved_game_points[i], normalized_game_points[i], total_game_steps[i], total_command_generation_f1[i]))
    print_strings.append("======================================================")
    print_strings = "\n".join(print_strings)
    print(print_strings)
    return np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), np.mean(total_command_generation_f1), print_strings


def evaluate_pretrained_command_generation(env, agent, valid_test="valid", verbose=False):
    env.split_reset(valid_test)
    agent.eval()
    total_soft_f1, total_exact_f1 = [], []
    counter = 0
    to_print = []

    while(True):
        observation_strings, triplets, target_strings = env.get_batch()
        pred_strings = agent.command_generation_greedy_generation(observation_strings, triplets)

        for i in range(len(observation_strings)):
            _, _, exact_f1 = get_match_result(pred_strings[i], target_strings[i], type='exact')
            _, _, soft_f1 = get_match_result(pred_strings[i], target_strings[i], type='soft')
            total_exact_f1.append(exact_f1)
            total_soft_f1.append(soft_f1)
            if verbose:
                to_print.append(str(counter) + " -------------------------------------------- exact f1: " + str(exact_f1) + ", soft f1: " + str(soft_f1))
                to_print.append("OBS: %s " % (observation_strings[i]))
                trips = []
                for t in triplets[i]:
                    trips.append(t[0] + "-" + t[2] + "-" + t[1])
                to_print.append("TRIPLETS: %s " % (" | ".join(trips)))
                to_print.append("PRED: %s " % (pred_strings[i]))
                to_print.append("GT: %s " % (target_strings[i]))
                to_print.append("")
            counter += 1
        if env.batch_pointer == 0:
            break
    if not agent.philly:
        with open(agent.experiment_tag + "_output.txt", "w") as f:
            f.write("\n".join(to_print))
    print("Hard F1: ", np.mean(np.array(total_exact_f1)), "Soft F1:", np.mean(np.array(total_soft_f1)))
    return np.mean(np.array(total_exact_f1)), np.mean(np.array(total_soft_f1))


def evaluate_action_prediction(env, agent, valid_test="valid", verbose=False):
    env.split_reset(valid_test)
    agent.eval()
    list_eval_acc, list_eval_loss = [], []
    counter = 0
    to_print = []

    while(True):
        current_graph, previous_graph, target_action, action_choices = env.get_batch()
        with torch.no_grad():
            loss, ap_ret, np_labels, action_choices = agent.get_action_prediction_logits(current_graph, previous_graph, target_action, action_choices)
        loss = to_np(loss)
        pred = np.argmax(ap_ret, -1)  # batch
        gt = np.argmax(np_labels, -1)  # batch
        correct = (pred == gt).astype("float32").tolist()
        list_eval_acc += correct
        list_eval_loss += [loss]

        if verbose:
            for i in range(len(current_graph)):
                to_print.append(str(counter) + " -------------------------------------------- acc: " + str(correct[i]))
                trips = []
                for t in previous_graph[i]:
                    trips.append(t[0] + "-" + t[2] + "-" + t[1])
                to_print.append("PREV TRIPLETS: %s " % (" | ".join(trips)))
                trips = []
                for t in current_graph[i]:
                    trips.append(t[0] + "-" + t[2] + "-" + t[1])
                to_print.append("CURR TRIPLETS: %s " % (" | ".join(trips)))
                to_print.append("PRED ACTION: %s " % (action_choices[i][pred[i]]))
                to_print.append("GT ACTION: %s " % (target_action[i]))
                to_print.append("")
                counter += 1

        if env.batch_pointer == 0:
            break
    if not agent.philly:
        with open(agent.experiment_tag + "_output.txt", "w") as f:
            f.write("\n".join(to_print))
    print("Eval Loss: {:2.3f}, Eval accuracy: {:2.3f}".format(np.mean(list_eval_loss), np.mean(list_eval_acc)))
    return np.mean(list_eval_loss), np.mean(list_eval_acc)


def evaluate_state_prediction(env, agent, valid_test="valid", verbose=False):
    env.split_reset(valid_test)
    agent.eval()
    list_eval_acc, list_eval_loss = [], []
    counter = 0
    to_print = []

    while(True):
        target_graph, previous_graph, action, admissible_graphs = env.get_batch()
        with torch.no_grad():
            loss, sp_ret, np_labels, admissible_graphs = agent.get_state_prediction_logits(previous_graph, action, target_graph, admissible_graphs)
        loss = to_np(loss)
        pred = np.argmax(sp_ret, -1)  # batch
        gt = np.argmax(np_labels, -1)  # batch
        correct = (pred == gt).astype("float32").tolist()
        list_eval_acc += correct
        list_eval_loss += [loss]

        if verbose:
            for i in range(len(previous_graph)):
                to_print.append(str(counter) + " -------------------------------------------- acc: " + str(correct[i]))
                trips = []
                for t in previous_graph[i]:
                    trips.append(t[0] + "-" + t[2] + "-" + t[1])
                to_print.append("PREV TRIPLETS: %s " % (" | ".join(trips)))
                to_print.append("ACTION: %s " % (action[i]))
                trips = []
                for t in admissible_graphs[i][pred[i]]:
                    trips.append(t[0] + "-" + t[2] + "-" + t[1])
                to_print.append("PRED TRIPLETS: %s " % (" | ".join(trips)))
                trips = []
                for t in target_graph[i]:
                    trips.append(t[0] + "-" + t[2] + "-" + t[1])
                to_print.append("GT TRIPLETS: %s " % (" | ".join(trips)))
                to_print.append("")
                counter += 1

        if env.batch_pointer == 0:
            break
    if not agent.philly:
        with open(agent.experiment_tag + "_output.txt", "w") as f:
            f.write("\n".join(to_print))
    print("Eval Loss: {:2.3f}, Eval accuracy: {:2.3f}".format(np.mean(list_eval_loss), np.mean(list_eval_acc)))
    return np.mean(list_eval_loss), np.mean(list_eval_acc)


def evaluate_deep_graph_infomax(env, agent, valid_test="valid", verbose=False):
    env.split_reset(valid_test)
    agent.eval()
    list_eval_acc, list_eval_loss = [], []
    # counter = 0
    # to_print = []

    while(True):
        triplets = env.get_batch()
        with torch.no_grad():
            loss, labels, dgi_discriminator_logits, batch_nonzero_idx = agent.get_deep_graph_infomax_logits(triplets)
        # sigmoid
        dgi_discriminator_logits = 1.0 / (1.0 + np.exp(-dgi_discriminator_logits))

        for i in range(len(triplets)):

            gt = labels[i]  # num_node*2
            pred_idx = (dgi_discriminator_logits[i] >= 0.5).astype("float32")  # num_node*2
            nonzeros = np.array(batch_nonzero_idx[i].tolist() + (batch_nonzero_idx[i] + len(agent.node_vocab)).tolist())
            gt = gt[nonzeros]  # num_nonzero
            pred_idx = pred_idx[nonzeros]  # num_nonzero
            correct = (pred_idx == gt).astype("float32").tolist()
            list_eval_acc += correct

        loss = to_np(loss)
        list_eval_loss.append(loss)

        if env.batch_pointer == 0:
            break
    return np.mean(list_eval_loss), np.mean(list_eval_acc)         
