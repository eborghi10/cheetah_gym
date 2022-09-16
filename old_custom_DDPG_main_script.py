import numpy as np
from multiprocessing import Pool
import argparse
from copy import deepcopy
import torch
import gym
from gym import wrappers

from cheetah_gym.envs.normalized_env import NormalizedEnv
from cheetah_gym.agents.evaluator import Evaluator
from cheetah_gym.agents.ddpg import DDPG
from cheetah_gym.agents.util import *
from old_custom_DDPG_train import train

# gym.undo_logger_setup()

if __name__ == "__main__":

    env = NormalizedEnv(gym.make('random1-v0'))

    seed = 14
    validate_steps = 4800
    max_episode_length = 240 * 20 # 4800 steps max per episode
    validate_episodes = 10
    train_iter = 300000

    if seed > 0:
        np.random.seed(seed)
        env.seed(seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, seed)
    evaluate = Evaluator(validate_episodes,
        validate_steps, "cheetah_gym/weights", max_episode_length=max_episode_length)

    train(train_iter, agent, env, evaluate, validate_steps, "cheetah_gym/weights", max_episode_length=max_episode_length, debug=True)
