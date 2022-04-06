from stable_baselines3 import DDPG, PPO, SAC
import numpy as np
import torch
import gym
from cheetah_gym.agents.util import *
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os

model_name = "PPO_normalized_obs_positive_distance"
models_dir = "models/" + model_name
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env("random1-v0", n_envs=16)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1000., clip_reward=1000.)
# env = VecNormalize.load(f"{models_dir}/env_normalization.pkl", env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
for i in range(3000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{models_dir}/{model.num_timesteps}")
    env.save(f"{models_dir}/env_normalization.pkl")


