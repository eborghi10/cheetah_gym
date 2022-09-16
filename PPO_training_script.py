from stable_baselines3 import DDPG, PPO, SAC
import numpy as np
import torch
import gym
from cheetah_gym.agents.util import *
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import os

model_name = "PPO_ZOO_PARAMS_5past_single_agent_distance_plus_3_speed_0_shared_layer"
models_dir = "models/" + model_name
logdir = "logs_new"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env_kwargs = {"visualize":True, "past_steps":5}
env = make_vec_env("random1-v0", n_envs=1, env_kwargs=env_kwargs)
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10000., clip_reward=10000.)
# env = VecNormalize.load(f"{models_dir}/env_normalization.pkl", env)

net_arch = [128, 128, dict(pi=[256, 256], vf=[256, 256])]
policy_kwargs = dict(activation_fn=torch.nn.ReLU, log_std_init=-1, ortho_init=False, net_arch=net_arch)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, gamma=0.99, batch_size=64, n_steps=512, gae_lambda=0.9,n_epochs=20, sde_sample_freq=4,
            use_sde=True, device='cuda', ent_coef=0.0, learning_rate=0.00003, clip_range=0.4, policy_kwargs=policy_kwargs)
# model.load(f"{models_dir}/{125337600}")

TIMESTEPS = 5000000
iters = 0
for i in range(3000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
    print("Saving model...")
    model.save(f"{models_dir}/{model.num_timesteps}")
    # env.save(f"{models_dir}/env_normalization.pkl")


