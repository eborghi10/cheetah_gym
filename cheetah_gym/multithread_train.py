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

# def train_single(agent):
#     env = gym.make('random1-v0')
#     nb_states = env.observation_space.shape[0]
#     nb_actions = env.action_space.shape[0]
#     agent_cpy = deepcopy(agent)
#     agent.is_training = True
#     step = episode = episode_steps = 0
#     episode_reward = 0.
#     observation = None
#     done = False
#     while not done:
#         # reset if it is the start of episode
#         if observation is None:
#             observation = deepcopy(env.reset())
#             agent.reset(observation)

#         # agent pick action ...
#         if step <= 100: #Warmup
#             action = agent.random_action()
#         else:
#             action = agent.select_action(observation)

#         # env response with next_observation, reward, terminate_info
#         observation2, reward, done, info = env.step(action)
#         observation2 = deepcopy(observation2)
#         if max_episode_length and episode_steps >= max_episode_length -1:
#             done = True

#         # agent observe and update policy
#         agent.observe(reward, observation2, done)
#         if step > 100 : #Warmup
#             agent.update_policy()

#         # update
#         step += 1
#         episode_steps += 1
#         episode_reward += reward
#         observation = deepcopy(observation2)

#     return [
#         observation,
#         agent.select_action(observation),
#         0.,
#         False]

# def train2(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
#     agent.is_training = True
#     step = episode = episode_steps = 0
#     last_step = 0
#     observation = None
#     -