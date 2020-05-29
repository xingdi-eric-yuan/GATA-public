import datetime
import os
import time
import copy
import json
import tempfile
import numpy as np
from os.path import join as pjoin
from distutils.dir_util import copy_tree
import torch

from agent import Agent
import generic
import evaluate
import reinforcement_learning_dataset
from generic import HistoryScoreCache, EpisodicCountingMemory


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    agent = Agent(config)
    output_dir = "."
    data_dir = "."

    # make game environments
    requested_infos = agent.select_additional_infos_lite()
    requested_infos_eval = agent.select_additional_infos()
    games_dir = "./"

    # training game env
    env, _ = reinforcement_learning_dataset.get_training_game_env(games_dir + config['rl']['data_path'],
                                                                  config['rl']['difficulty_level'],
                                                                  config['rl']['training_size'],
                                                                  requested_infos,
                                                                  agent.max_nb_steps_per_episode,
                                                                  agent.batch_size)

    if agent.run_eval:
        # training game env
        eval_env, num_eval_game = reinforcement_learning_dataset.get_evaluation_game_env(games_dir + config['rl']['data_path'],
                                                                                         config['rl']['difficulty_level'],
                                                                                         requested_infos_eval,
                                                                                         agent.eval_max_nb_steps_per_episode,
                                                                                         agent.eval_batch_size,
                                                                                         valid_or_test="valid")
    else:
        eval_env, num_eval_game = None, None

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        reward_win, step_win = None, None
        dqn_loss_win = None
        eval_game_points_win, eval_step_win = None, None
        viz_game_rewards, viz_game_points, viz_game_points_normalized, viz_graph_rewards, viz_count_rewards, viz_step = [], [], [], [], [], []
        viz_dqn_loss = []
        viz_eval_game_points, viz_eval_game_points_normalized, viz_eval_step = [], [], []

    step_in_total = 0
    episode_no = 0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_game_points_normalized = HistoryScoreCache(capacity=500)
    running_avg_graph_rewards = HistoryScoreCache(capacity=500)
    running_avg_count_rewards = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)
    running_avg_dqn_loss = HistoryScoreCache(capacity=500)
    running_avg_game_rewards = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_train_performance_so_far, best_eval_performance_so_far = 0.0, 0.0
    prev_performance = 0.0

    if os.path.exists(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt"):
        agent.load_pretrained_graph_generation_model(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt")
    else:
        print("No real-valued graph generation module detected... Please check ", data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt") 

    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            # this experiment itself (in case the experiment crashes for unknown reasons on server)
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
            agent.update_target_net()
        elif os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            # load from pre-trained graph encoder
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    i_am_patient = 0
    perfect_training = 0
    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        # filter look and examine actions
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        batch_size = len(obs)

        agent.train()
        agent.init()

        game_name_list = [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list = [game.max_score for game in infos["game"]]
        chosen_actions = []
        prev_step_dones, prev_rewards = [], []
        prev_graph_hidden_state = torch.zeros(batch_size, agent.online_net.block_hidden_dim)
        if agent.use_cuda:
            prev_graph_hidden_state = prev_graph_hidden_state.cuda()
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)

        prev_h, prev_c = None, None
        episodes_masks = 1 - torch.tensor(prev_step_dones) # inverse of `prev_step_dones`
        episodes_masks = episodes_masks.cuda() if agent.use_cuda else episodes_masks

        observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite(obs, infos)
        observation_for_counting = copy.copy(observation_strings)

        if agent.count_reward_lambda > 0:
            agent.reset_binarized_counter(batch_size)
            _ = agent.get_binarized_count(observation_for_counting)

        # it requires to store sequences of transitions into memory with order,
        # so we use a cache to keep what agents returns, and push them into memory
        # altogether in the end of game.
        transition_cache = []
        still_running_mask = []
        game_rewards, game_points, graph_rewards, count_rewards = [], [], [], []
        print_actions = []

        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode
        for step_no in range(agent.max_nb_steps_per_episode):
            if agent.noisy_net:
                agent.reset_noise()  # Draw a new set of noisy weights

            # generate adj_matrices
            new_adjacency_matrix, new_graph_hidden_state = agent.generate_adjacency_matrix_for_rl(observation_strings, chosen_actions, prev_graph_hidden_state)
            new_chosen_actions, chosen_indices, prev_h, prev_c = agent.act(observation_strings, new_adjacency_matrix, action_candidate_list, previous_h=prev_h, previous_c=prev_c, random=act_randomly)
            replay_info = [observation_strings, action_candidate_list, chosen_indices, generic.to_np(prev_graph_hidden_state), chosen_actions]
            transition_cache.append(replay_info)
            chosen_actions = new_chosen_actions
            chosen_actions_before_parsing =  [item[idx] for item, idx in zip(infos["admissible_commands"], chosen_indices)]
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            # filter look and examine actions
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)
            ## prev_triplets = current_triplets # commented for obs_gen
            prev_graph_hidden_state = new_graph_hidden_state
            observation_strings, action_candidate_list = agent.get_game_info_at_certain_step_lite(obs, infos)
            observation_for_counting = copy.copy(observation_strings)

            if agent.noisy_net and step_in_total % agent.update_per_k_game_steps == 0:
                agent.reset_noise()  # Draw a new set of noisy weights
            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                dqn_loss, _ = agent.update_dqn(episode_no)
                if dqn_loss is not None:
                    running_avg_dqn_loss.push(dqn_loss)

            if step_no == agent.max_nb_steps_per_episode - 1:
                # terminate the game because DQN requires one extra step
                dones = [True for _ in dones]

            step_in_total += 1
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
            game_points.append(copy.copy(step_rewards))
            if agent.use_negative_reward:
                step_rewards = [-1.0 if _lost else r for r, _lost in zip(step_rewards, infos["has_lost"])]  # list of float
                step_rewards = [5.0 if _won else r for r, _won in zip(step_rewards, infos["has_won"])]  # list of float
            prev_rewards = scores
            step_graph_rewards = [0.0 for _ in range(batch_size)] ## adding for obs_gen
            # counting bonus
            if agent.count_reward_lambda > 0:
                step_revisit_counting_rewards = agent.get_binarized_count(observation_for_counting, update=True)
                step_revisit_counting_rewards = [r * agent.count_reward_lambda for r in step_revisit_counting_rewards]
            else:
                step_revisit_counting_rewards = [0.0 for _ in range(batch_size)]
            still_running_mask.append(still_running)
            game_rewards.append(step_rewards)
            graph_rewards.append(step_graph_rewards)
            count_rewards.append(step_revisit_counting_rewards)
            print_actions.append(chosen_actions_before_parsing[0] if still_running[0] else "--")

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        still_running_mask_np = np.array(still_running_mask)
        game_rewards_np = np.array(game_rewards) * still_running_mask_np  # step x batch
        game_points_np = np.array(game_points) * still_running_mask_np  # step x batch
        graph_rewards_np = np.array(graph_rewards) * still_running_mask_np  # step x batch
        count_rewards_np = np.array(count_rewards) * still_running_mask_np  # step x batch
        if agent.graph_reward_lambda > 0.0:
            graph_rewards_pt = generic.to_pt(graph_rewards_np, enable_cuda=agent.use_cuda, type='float')  # step x batch
        else:
            graph_rewards_pt = generic.to_pt(np.zeros_like(graph_rewards_np), enable_cuda=agent.use_cuda, type='float')  # step x batch
        if agent.count_reward_lambda > 0.0:
            count_rewards_pt = generic.to_pt(count_rewards_np, enable_cuda=agent.use_cuda, type='float')  # step x batch
        else:
            count_rewards_pt = generic.to_pt(np.zeros_like(count_rewards_np), enable_cuda=agent.use_cuda, type='float')  # step x batch
        command_rewards_pt = generic.to_pt(game_rewards_np, enable_cuda=agent.use_cuda, type='float')  # step x batch

        # push experience into replay buffer (dqn)
        avg_rewards_in_buffer = agent.dqn_memory.avg_rewards()
        for b in range(game_rewards_np.shape[1]):
            if still_running_mask_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_np[-1][b] != 0:
                # need to pad one transition
                _need_pad = True
                tmp_game_rewards = game_rewards_np[:, b].tolist() + [0.0]
            else:
                _need_pad = False
                tmp_game_rewards = game_rewards_np[:, b]
            if np.mean(tmp_game_rewards) < avg_rewards_in_buffer * agent.buffer_reward_threshold:
                continue
            for i in range(game_rewards_np.shape[0]):
                observation_strings, action_candidate_list, chosen_indices, graph_hidden_state, prev_action_strings = transition_cache[i]
                is_final = True
                if still_running_mask_np[i][b] != 0:
                    is_final = False
                agent.dqn_memory.add(observation_strings[b], prev_action_strings[b], action_candidate_list[b], chosen_indices[b], graph_hidden_state[b], command_rewards_pt[i][b], graph_rewards_pt[i][b], count_rewards_pt[i][b], is_final)
                if still_running_mask_np[i][b] == 0:
                    break
            if _need_pad:
                observation_strings, action_candidate_list, chosen_indices, graph_hidden_state, prev_action_strings = transition_cache[-1]
                agent.dqn_memory.add(observation_strings[b], prev_action_strings[b], action_candidate_list[b], chosen_indices[b], graph_hidden_state[b], command_rewards_pt[-1][b] * 0.0, graph_rewards_pt[-1][b] * 0.0, count_rewards_pt[-1][b] * 0.0, True)

        for b in range(batch_size):
            running_avg_game_points.push(np.sum(game_points_np, 0)[b])
            game_max_score_np = np.array(game_max_score_list, dtype="float32")
            running_avg_game_points_normalized.push((np.sum(game_points_np, 0) / game_max_score_np)[b])
            running_avg_game_steps.push(np.sum(still_running_mask_np, 0)[b])
            running_avg_game_rewards.push(np.sum(game_rewards_np, 0)[b])
            running_avg_graph_rewards.push(np.sum(graph_rewards_np, 0)[b])
            running_avg_count_rewards.push(np.sum(count_rewards_np, 0)[b])

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size

        if episode_no < agent.learn_start_from_this_episode:
            continue
        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - batch_size) % agent.report_frequency):
            continue
        time_2 = datetime.datetime.now()
        print("Episode: {:3d} | time spent: {:s} | dqn loss: {:2.3f} | game points: {:2.3f} | normalized game points: {:2.3f} | game rewards: {:2.3f} | graph rewards: {:2.3f} | count rewards: {:2.3f} | used steps: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], running_avg_dqn_loss.get_avg(), running_avg_game_points.get_avg(), running_avg_game_points_normalized.get_avg(), running_avg_game_rewards.get_avg(), running_avg_graph_rewards.get_avg(), running_avg_count_rewards.get_avg(), running_avg_game_steps.get_avg()))
        print(game_name_list[0] + ":    " + " | ".join(print_actions))

        # evaluate
        curr_train_performance = running_avg_game_points_normalized.get_avg()
        eval_game_points, eval_game_points_normalized, eval_game_step = 0.0, 0.0, 0.0
        if agent.run_eval:
            eval_game_points, eval_game_points_normalized, eval_game_step, detailed_scores = evaluate.evaluate_rl_with_real_graphs(eval_env, agent, num_eval_game)
            curr_eval_performance = eval_game_points_normalized
            curr_performance = curr_eval_performance
            if curr_eval_performance > best_eval_performance_so_far:
                best_eval_performance_so_far = curr_eval_performance
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
            elif curr_eval_performance == best_eval_performance_so_far:
                if curr_eval_performance > 0.0:
                    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                else:
                    if curr_train_performance >= best_train_performance_so_far:
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
        else:
            curr_eval_performance = 0.0
            detailed_scores = ""
            curr_performance = curr_train_performance
            if curr_train_performance >= best_train_performance_so_far:
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
        # update best train performance
        if curr_train_performance >= best_train_performance_so_far:
            best_train_performance_so_far = curr_train_performance

        if prev_performance <= curr_performance:
            i_am_patient = 0
        else:
            i_am_patient += 1
        prev_performance = curr_performance

        # if patient >= patience, resume from checkpoint
        if agent.patience > 0 and i_am_patient >= agent.patience:
            if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
                print('reload from a good checkpoint...')
                agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
                agent.update_target_net()
                i_am_patient = 0

        if running_avg_game_points_normalized.get_avg() >= 0.95:
            perfect_training += 1
        else:
            perfect_training = 0

        # plot using visdom
        if config["general"]["visdom"]:
            viz_game_rewards.append(running_avg_game_rewards.get_avg())
            viz_game_points.append(running_avg_game_points.get_avg())
            viz_game_points_normalized.append(running_avg_game_points_normalized.get_avg())
            viz_graph_rewards.append(running_avg_graph_rewards.get_avg())
            viz_count_rewards.append(running_avg_count_rewards.get_avg())
            viz_step.append(running_avg_game_steps.get_avg())
            viz_dqn_loss.append(running_avg_dqn_loss.get_avg())
            viz_eval_game_points.append(eval_game_points)
            viz_eval_game_points_normalized.append(eval_game_points_normalized)
            viz_eval_step.append(eval_game_step)
            viz_x = np.arange(len(viz_game_rewards)).tolist()

            if reward_win is None:
                reward_win = viz.line(X=viz_x, Y=viz_game_rewards,
                                   opts=dict(title=agent.experiment_tag + "_game_rewards"),
                                   name="game_rewards")
                viz.line(X=viz_x, Y=viz_graph_rewards,
                         opts=dict(title=agent.experiment_tag + "_graph_rewards"),
                         win=reward_win, update='append', name="graph_rewards")
                viz.line(X=viz_x, Y=viz_count_rewards,
                         opts=dict(title=agent.experiment_tag + "_count_rewards"),
                         win=reward_win, update='append', name="count_rewards")
                viz.line(X=viz_x, Y=viz_game_points,
                         opts=dict(title=agent.experiment_tag + "_game_points"),
                         win=reward_win, update='append', name="game_points")
                viz.line(X=viz_x, Y=viz_game_points_normalized,
                         opts=dict(title=agent.experiment_tag + "_game_points_normalized"),
                         win=reward_win, update='append', name="game_points_normalized")
            else:
                viz.line(X=[len(viz_game_rewards) - 1], Y=[viz_game_rewards[-1]],
                         opts=dict(title=agent.experiment_tag + "_game_rewards"),
                         win=reward_win,
                         update='append', name="game_rewards")
                viz.line(X=[len(viz_graph_rewards) - 1], Y=[viz_graph_rewards[-1]],
                         opts=dict(title=agent.experiment_tag + "_graph_rewards"),
                         win=reward_win,
                         update='append', name="graph_rewards")
                viz.line(X=[len(viz_count_rewards) - 1], Y=[viz_count_rewards[-1]],
                         opts=dict(title=agent.experiment_tag + "_count_rewards"),
                         win=reward_win,
                         update='append', name="count_rewards")
                viz.line(X=[len(viz_game_points) - 1], Y=[viz_game_points[-1]],
                         opts=dict(title=agent.experiment_tag + "_game_points"),
                         win=reward_win,
                         update='append', name="game_points")
                viz.line(X=[len(viz_game_points_normalized) - 1], Y=[viz_game_points_normalized[-1]],
                         opts=dict(title=agent.experiment_tag + "_game_points_normalized"),
                         win=reward_win,
                         update='append', name="game_points_normalized")

            if step_win is None:
                step_win = viz.line(X=viz_x, Y=viz_step,
                                   opts=dict(title=agent.experiment_tag + "_step"),
                                   name="step")
            else:
                viz.line(X=[len(viz_step) - 1], Y=[viz_step[-1]],
                         opts=dict(title=agent.experiment_tag + "_step"),
                         win=step_win,
                         update='append', name="step")

            if dqn_loss_win is None:
                dqn_loss_win = viz.line(X=viz_x, Y=viz_dqn_loss,
                                   opts=dict(title=agent.experiment_tag + "_dqn_loss"),
                                   name="dqn loss")
            else:
                viz.line(X=[len(viz_dqn_loss) - 1], Y=[viz_dqn_loss[-1]],
                         opts=dict(title=agent.experiment_tag + "_dqn_loss"),
                         win=dqn_loss_win,
                         update='append', name="dqn loss")

            if eval_game_points_win is None:
                eval_game_points_win = viz.line(X=viz_x, Y=viz_eval_game_points,
                                   opts=dict(title=agent.experiment_tag + "_eval_game_points"),
                                   name="eval game points")
                viz.line(X=viz_x, Y=viz_eval_game_points_normalized,
                         opts=dict(title=agent.experiment_tag + "_eval_game_points_normalized"),
                         win=eval_game_points_win, update='append', name="eval_game_points_normalized")
            else:
                viz.line(X=[len(viz_eval_game_points) - 1], Y=[viz_eval_game_points[-1]],
                         opts=dict(title=agent.experiment_tag + "_eval_game_points"),
                         win=eval_game_points_win,
                         update='append', name="eval game_points")
                viz.line(X=[len(viz_eval_game_points_normalized) - 1], Y=[viz_eval_game_points_normalized[-1]],
                         opts=dict(title=agent.experiment_tag + "_eval_game_points_normalized"),
                         win=eval_game_points_win,
                         update='append', name="eval_game_points_normalized")

            if eval_step_win is None:
                eval_step_win = viz.line(X=viz_x, Y=viz_eval_step,
                                   opts=dict(title=agent.experiment_tag + "_eval_step"),
                                   name="eval step")
            else:
                viz.line(X=[len(viz_eval_step) - 1], Y=[viz_eval_step[-1]],
                         opts=dict(title=agent.experiment_tag + "_eval_step"),
                         win=eval_step_win,
                         update='append', name="eval step")

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "dqn loss": str(running_avg_dqn_loss.get_avg()),
                         "train game points": str(running_avg_game_points.get_avg()),
                         "train normalized game points": str(running_avg_game_points_normalized.get_avg()),
                         "train game rewards": str(running_avg_game_rewards.get_avg()),
                         "train graph rewards": str(running_avg_graph_rewards.get_avg()),
                         "train count rewards": str(running_avg_count_rewards.get_avg()),
                         "train steps": str(running_avg_game_steps.get_avg()),
                         "eval game points": str(eval_game_points),
                         "eval normalized game points": str(eval_game_points_normalized),
                         "eval steps": str(eval_game_step),
                         "detailed scores": detailed_scores})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()

        if curr_performance == 1.0 and curr_train_performance >= 0.95:
            break
        if perfect_training >= 3:
            break


if __name__ == '__main__':
    train()
