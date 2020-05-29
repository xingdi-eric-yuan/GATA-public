import torch
import datetime
import os
import time
import json
import math
import numpy as np
from os.path import join as pjoin

from observation_generation_dataset import ObservationGenerationData
from agent import Agent
import generic
import evaluate


def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    env = ObservationGenerationData(config)
    env.split_reset("train")
    agent = Agent(config)
    agent.zero_noise()
    ave_train_loss = generic.HistoryScoreCache(capacity=500)

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        plt_win = None
        viz_loss, viz_eval_loss = [], []

    episode_no = 0
    batch_no = 0

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_training_loss_so_far, best_eval_loss_so_far = 10000.0, 10000.0
    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
        elif os.path.exists(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_graph_generation_model_from_tag + ".pt", load_partial_graph=False)

    try:
        while(True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            observation_strings, prev_action_strings = env.get_batch()
            training_losses, _ = agent.get_observation_infomax_loss(observation_strings, prev_action_strings)

            curr_batch_size = len(observation_strings)
            for _loss in training_losses:
                ave_train_loss.push(_loss)

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

            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], ave_train_loss.get_avg()))

            if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - curr_batch_size) % agent.report_frequency):
                continue

            eval_loss, eval_acc = 100000.0, 0
            if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
                if agent.run_eval:
                    eval_loss, eval_acc = evaluate.evaluate_observation_infomax(env, agent, "valid")
                    env.split_reset("train")
                    # if run eval, then save model by eval accuracy
                    if eval_loss < best_eval_loss_so_far:
                        best_eval_loss_so_far = eval_loss
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                else:
                    loss = ave_train_loss.get_avg()
                    if loss < best_training_loss_so_far:
                        best_training_loss_so_far = loss
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")

            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f} | valid loss: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], ave_train_loss.get_avg(), eval_loss))

            # plot using visdom
            if config["general"]["visdom"]:
                viz_loss.append(ave_train_loss.get_avg())
                viz_eval_loss.append(eval_loss)
                viz_x = np.arange(len(viz_loss)).tolist()

                if plt_win is None:
                    plt_win = viz.line(X=viz_x, Y=viz_loss,
                                    opts=dict(title=agent.experiment_tag + "_loss"),
                                    name="training loss")

                    viz.line(X=viz_x, Y=viz_eval_loss,
                            opts=dict(title=agent.experiment_tag + "_eval_loss"),
                            win=plt_win,
                            update='append', name="eval loss")
                else:
                    viz.line(X=[len(viz_loss) - 1], Y=[viz_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_loss"),
                            win=plt_win,
                            update='append', name="training loss")

                    viz.line(X=[len(viz_eval_loss) - 1], Y=[viz_eval_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval_loss"),
                            win=plt_win,
                            update='append', name="eval loss")

            # write accuracies down into file
            _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                            "loss": str(ave_train_loss.get_avg()),
                            "eval loss": str(eval_loss),
                            "eval accuracy": str(eval_acc)})
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
        eval_loss, eval_acc = evaluate.evaluate_observation_infomax(env, agent, "test")


if __name__ == '__main__':
    train()
