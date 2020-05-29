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
        eval_plt_win = None
        viz_loss, viz_eval_loss, viz_eval_f1 = [], [], []

    episode_no = 0
    batch_no = 0

    output_dir = "."
    data_dir = "."
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_eval_loss_so_far, best_training_loss_so_far = 10000.0, 10000.0
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
            curr_batch_size = len(observation_strings)
            lens = [len(elem) for elem in observation_strings]
            max_len = max(lens)
            padded_observation_strings = [elem + ["<pad>"]*(max_len - len(elem)) for elem in observation_strings]
            padded_prev_action_strings = [elem + ["<pad>"]*(max_len - len(elem)) for elem in prev_action_strings]
            masks = torch.zeros((curr_batch_size, max_len), dtype=torch.float).cuda() if agent.use_cuda else torch.zeros((curr_batch_size, max_len), dtype=torch.float)
            for i in range(curr_batch_size):
                masks[i, :lens[i]] = 1
            preds_last_batch = []
            last_k_batches_loss = []
            prev_h = None
            for i in range(max_len):
                batch_obs_string = [elem[i] for elem in padded_observation_strings]
                batch_prev_action_string = [elem[i] for elem in padded_prev_action_strings]
                loss, pred, prev_h = agent.observation_generation_teacher_force(batch_obs_string, batch_prev_action_string, masks[:, i], prev_h)
                last_k_batches_loss.append(loss)
                ave_train_loss.push(generic.to_np(loss))
                preds_last_batch.append(pred[-1])
                if ((i + 1) % agent.backprop_frequency == 0 or i == max_len - 1):  # and i > 0:
                    agent.optimizer.zero_grad()
                    ave_k_loss = torch.mean(torch.stack(last_k_batches_loss))
                    ave_k_loss.backward()
                    agent.optimizer.step()
                    last_k_batches_loss = []
                    prev_h = prev_h.detach()

            k = 0
            ep_string = []
            while(masks[-1][k] > 0):
                step_string = []
                regen_strings = preds_last_batch[k].argmax(-1)
                for l in range(len(regen_strings)):
                    step_string.append(agent.word_vocab[regen_strings[l]])
                ep_string.append((' '.join(step_string).split("<eos>")[0]))
                k += 1
                if k == len(masks[-1]):
                    break
            if len(ep_string) >= 3:
                print(' | '.join(ep_string[:3]))
            #####

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

            eval_loss, eval_f1 = 0.0, 0.0
            if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
                if agent.run_eval:
                    eval_loss = evaluate.evaluate_observation_generation_loss(env, agent, "valid")
                    eval_f1 = evaluate.evaluate_observation_generation_free_generation(env, agent, "valid")
                    env.split_reset("train")
                    # if run eval, then save model by eval accuracy
                    if eval_loss < best_eval_loss_so_far:
                        best_eval_loss_so_far = eval_loss
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                else:
                    if loss < best_training_loss_so_far:
                        best_training_loss_so_far = loss
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")


            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | loss: {:2.3f} | valid loss: {:2.3f} | valid f1: {:2.3f}".format(episode_no, str(time_2 - time_1).rsplit(".")[0], loss, eval_loss, eval_f1))

            # plot using visdom
            if config["general"]["visdom"]:
                viz_loss.append(ave_train_loss.get_avg())
                viz_eval_loss.append(eval_loss)
                viz_eval_f1.append(eval_f1)
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


                if eval_plt_win is None:
                    eval_plt_win = viz.line(X=viz_x, Y=viz_eval_f1,
                                   opts=dict(title=agent.experiment_tag + "_eval_f1"),
                                   name="eval f1")
                else:
                    viz.line(X=[len(viz_eval_f1) - 1], Y=[viz_eval_f1[-1]],
                            opts=dict(title=agent.experiment_tag + "_eval_f1"),
                            win=eval_plt_win,
                            update='append', name="eval f1")

            # write accuracies down into file
            _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                            "loss": str(ave_train_loss.get_avg()),
                            "eval loss": str(eval_loss),
                            "eval f1": str(eval_f1)})
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
        test_loss = evaluate.evaluate_observation_generation_loss(env, agent, "test")
        test_f1 = evaluate.evaluate_observation_generation_free_generation(env, agent, "test")
        print(test_loss, test_f1)


if __name__ == '__main__':
    train()
