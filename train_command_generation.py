import datetime
import os
import time
import json
import math
import numpy as np
from os.path import join as pjoin

from command_generation_dataset import CommandGenerationData
from agent import Agent
import generic
import evaluate


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    env = CommandGenerationData(config)
    env.split_reset("train")
    agent = Agent(config)
    agent.zero_noise()
    ave_train_loss = generic.HistoryScoreCache(capacity=500)

    # visdom
    import visdom
    viz = visdom.Visdom()
    plt_win = None
    eval_plt_win = None
    viz_loss, viz_eval_exact_f1, viz_eval_soft_f1 = [], [], []

    episode_no = 0
    batch_no = 0

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_eval_exact_f1_so_far, best_eval_soft_f1_so_far, best_training_loss_so_far = 0.0, 0.0, 10000.0
    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
        elif os.path.exists(data_dir + "/" + agent.load_graph_update_model_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_graph_update_model_from_tag + ".pt", load_partial_graph=False)

    try:
        while(True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            observation_strings, triplets, target_strings = env.get_batch()
            curr_batch_size = len(observation_strings)

            _, loss = agent.command_generation_teacher_force(observation_strings, triplets, target_strings)
            ave_train_loss.push(loss)

            # lr schedule
            # learning_rate = 1.0 * (generic.power(agent.model.block_hidden_dim, -0.5) * min(generic.power(batch_no, -0.5), batch_no * generic.power(agent.learning_rate_warmup_until, -1.5)))
            if batch_no < agent.learning_rate_warmup_until:
                cr = agent.init_learning_rate / math.log2(agent.learning_rate_warmup_until)
                learning_rate = cr * math.log2(batch_no + 1)
            else:
                learning_rate = agent.init_learning_rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = learning_rate

            episode_no += curr_batch_size
            batch_no += 1

            if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - curr_batch_size) % agent.report_frequency):
                continue

            viz_loss.append(ave_train_loss.get_avg())

            eval_f1_exact, eval_f1_soft = 0.0, 0.0
            if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
                if agent.run_eval:
                    eval_f1_exact, eval_f1_soft = evaluate.evaluate_pretrained_command_generation(env, agent, "valid")
                    env.split_reset("train")
                    # if run eval, then save model by eval accuracy
                    if eval_f1_exact > best_eval_exact_f1_so_far:
                        best_eval_exact_f1_so_far = eval_f1_exact
                        best_eval_soft_f1_so_far = eval_f1_soft
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                    elif eval_f1_exact == best_eval_exact_f1_so_far and eval_f1_soft > best_eval_soft_f1_so_far:
                        best_eval_soft_f1_so_far = eval_f1_soft
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                else:
                    if loss < best_training_loss_so_far:
                        best_training_loss_so_far = loss
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")

            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f} | valid exact f1: {:2.3f} | valid soft f1: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], loss, eval_f1_exact, eval_f1_soft))

            # plot using visdom
            viz_eval_exact_f1.append(eval_f1_exact)
            viz_eval_soft_f1.append(eval_f1_soft)
            viz_x = np.arange(len(viz_loss)).tolist()
            viz_eval_x = np.arange(len(viz_eval_exact_f1)).tolist()

            if plt_win is None:
                plt_win = viz.line(X=viz_x, Y=viz_loss,
                                opts=dict(title=agent.experiment_tag + "_loss"),
                                name="training loss")
            else:
                viz.line(X=[len(viz_loss) - 1], Y=[viz_loss[-1]],
                        opts=dict(title=agent.experiment_tag + "_loss"),
                        win=plt_win,
                        update='append', name="training loss")

            if eval_plt_win is None:
                eval_plt_win = viz.line(X=viz_eval_x, Y=viz_eval_exact_f1,
                                opts=dict(title=agent.experiment_tag + "_exact_f1"),
                                name="eval exact f1")
                viz.line(X=viz_eval_x, Y=viz_eval_soft_f1,
                        opts=dict(title=agent.experiment_tag + "_soft_f1"),
                        win=eval_plt_win, update='append', name="eval soft f1")
            else:
                viz.line(X=[len(viz_eval_exact_f1) - 1], Y=[viz_eval_exact_f1[-1]],
                        opts=dict(title=agent.experiment_tag + "_exact_f1"),
                        win=eval_plt_win,
                        update='append', name="eval exact f1")
                viz.line(X=[len(viz_eval_soft_f1) - 1], Y=[viz_eval_soft_f1[-1]],
                        opts=dict(title=agent.experiment_tag + "_soft_f1"),
                        win=eval_plt_win,
                        update='append', name="eval soft f1")

            # write accuracies down into file
            _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                            "loss": str(ave_train_loss.get_avg()),
                            "eval exact f1": str(eval_f1_exact),
                            "eval soft f1": str(eval_f1_soft)})
            with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
                outfile.write(_s + '\n')
                outfile.flush()
    
    # At any point you can hit Ctrl + C to break out of training early.
    except KeyboardInterrupt:
        print('--------------------------------------------')
        print('Exiting from training early...')
    if agent.run_eval:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            print('Evaluating on test set and saving log...')
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
        _, _ = evaluate.evaluate_pretrained_command_generation(env, agent, "test", verbose=True)


if __name__ == '__main__':
    train()