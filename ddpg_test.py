from stable_baselines3 import DDPG, PPO, SAC
import numpy as np
import torch
import gym
from cheetah_gym.agents.util import *
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("random1-v0", n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
models_dir = "models/PPO_normalized_obs"

model = PPO("MlpPolicy", env, verbose=1)

model.load(f"{models_dir}/{2998272}")
env = VecNormalize.load(f"{models_dir}/env_normalization.pkl", env)
env.training = False
env.norm_reward = False

# input("press enter to continue...")
obs = env.reset()
lifes = 10
while lifes > 0:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        # plt.imshow(env.render())
        # plt.show()
        if done:
            input("press enter to continue...")
            obs = env.reset()
            break
    lifes -= 1
