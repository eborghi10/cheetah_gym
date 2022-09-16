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

def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    # agent.load_weights(output)
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    last_step = 0
    observation = None
    max_reward = -np.inf
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= 1000: #Warmup
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > 100 : #Warmup
            agent.update_policy()

        # # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            # monitor_env = wrappers.Monitor(env, output + '/video/' + str(step), video_callable=lambda episode_id: True, force=True)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow(f'[Evaluate] Step_{step:07d}: mean_reward:{validate_reward}')
            # monitor_env.close()

        # [optional] save intermideate model
        if episode % int(5) == 0:
            agent.save_model(output)
        
        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step - last_step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            if episode_reward > max_reward:
                max_reward = episode_reward
                agent.save_model(output + '/best')
            episode_reward = 0.
            episode += 1
            last_step = step

    print('max_reward:{}'.format(max_reward))