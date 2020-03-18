import os
import glob
import gym
import textworld.gym


def get_training_game_env(data_dir, difficulty_level, training_size, requested_infos, max_episode_steps, batch_size):
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert training_size in [1, 20, 100]

    # training games
    game_file_names = []
    game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)

    env_id = textworld.gym.register_games(sorted(game_file_names), request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="training", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names)
    return env, num_game


def get_evaluation_game_env(data_dir, difficulty_level, requested_infos, max_episode_steps, batch_size, valid_or_test="valid"):
    assert valid_or_test in ["valid", "test"]
    assert difficulty_level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # eval games
    game_file_names = []
    game_path = data_dir + "/" + valid_or_test + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)

    env_id = textworld.gym.register_games(sorted(game_file_names), request_infos=requested_infos,
                                          max_episode_steps=max_episode_steps, batch_size=batch_size,
                                          name="eval", asynchronous=True, auto_reset=False)
    env = gym.make(env_id)
    num_game = len(game_file_names)
    return env, num_game
