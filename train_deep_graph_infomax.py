import datetime
import os
import time
import json
import math
import numpy as np
import torch
from os.path import join as pjoin

from deep_graph_infomax_dataset import DGIData
from agent import Agent
import generic
import evaluate
from generic import to_pt, to_np

def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    env = DGIData(config)
    env.split_reset("train")
    agent = Agent(config)
    agent.zero_noise()
    ave_train_loss = generic.HistoryScoreCache(capacity=500)

    # visdom
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        loss_win = None
        eval_acc_win = None
        viz_loss, viz_eval_loss, viz_eval_acc = [], [], []

    episode_no = 0
    batch_no = 0

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    ####best_eval_exact_f1_so_far, best_eval_soft_f1_so_far = 0.0, 0.0
    # load model from checkpoint
    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
        elif os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt", load_partial_graph=False)

    best_eval_acc, best_training_loss_so_far = 0.0, 10000.0

    try:
        while(True):
            if episode_no > agent.max_episode:
                break
            agent.train()
            triplets = env.get_batch()
            curr_batch_size = len(triplets)
            loss, _, _, _ = agent.get_deep_graph_infomax_logits(triplets)
            # Update Model
            agent.online_net.zero_grad()
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), agent.clip_grad_norm)
            agent.optimizer.step()
            loss = generic.to_np(loss)
            ave_train_loss.push(loss)

            # lr schedule
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

            eval_acc, eval_loss = 0.0, 0.0
            if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
                if agent.run_eval:
                    eval_loss, eval_acc = evaluate.evaluate_deep_graph_infomax(env, agent, "valid")
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                        print("Saving best model so far! with Eval acc : {:2.3f}".format(best_eval_acc))
                    env.split_reset("train")
                else:
                    if loss < best_training_loss_so_far:
                        best_training_loss_so_far = loss
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")

            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | sliding window loss: {:2.3f} | Eval Acc: {:2.3f} | Eval Loss: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], ave_train_loss.get_avg(), eval_acc, eval_loss))

            # plot using visdom
            if config["general"]["visdom"]:
                viz_loss.append(ave_train_loss.get_avg())
                viz_eval_acc.append(eval_acc)
                viz_eval_loss.append(eval_loss)
                viz_x = np.arange(len(viz_loss)).tolist()
                viz_eval_x = np.arange(len(viz_eval_acc)).tolist()

                if loss_win is None:
                    loss_win = viz.line(X=viz_x, Y=viz_loss,
                                        opts=dict(title=agent.experiment_tag + "_loss"),
                                        name="training loss")
                    viz.line(X=viz_eval_x, Y=viz_eval_loss,
                            opts=dict(title=agent.experiment_tag + "_eval_loss"),
                            win=loss_win, update='append', name="eval loss")
                else:
                    viz.line(X=[len(viz_loss) - 1], Y=[viz_loss[-1]],
                                opts=dict(title=agent.experiment_tag + "_loss"),
                                win=loss_win,
                                update='append', name="training loss")
                    viz.line(X=[len(viz_eval_loss) - 1], Y=[viz_eval_loss[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval_loss"),
                            win=loss_win, update='append', name="eval loss")

                if eval_acc_win is None:
                    eval_acc_win = viz.line(X=viz_eval_x, Y=viz_eval_acc,
                                            opts=dict(title=agent.experiment_tag + "_eval_acc"),
                                            name="eval accuracy")
                else:
                    viz.line(X=[len(viz_eval_acc) - 1], Y=[viz_eval_acc[-1]],
                                opts=dict(title=agent.experiment_tag + "_eval_acc"),
                                win=eval_acc_win,
                                update='append', name="eval accuracy")

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
        _, _ = evaluate.evaluate_deep_graph_infomax(env, agent, "test", verbose=True)


if __name__ == '__main__':
    train()
