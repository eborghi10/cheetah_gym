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

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

if __name__ == "__main__":

    env = NormalizedEnv(gym.make('random1-v0', visualize=True))

    seed = 42
    validate_steps = 2000
    max_episode_length = 60 * 15
    validate_episodes = 20

    if seed > 0:
        np.random.seed(seed)
        env.seed(seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, seed)
    evaluate = Evaluator(validate_episodes,
        validate_steps, "cheetah_gym/weights/3M iter", max_episode_length=max_episode_length)

    test(validate_episodes, agent, env, evaluate, "cheetah_gym/weights/3M iter", visualize=False, debug=True)