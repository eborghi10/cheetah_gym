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

# gym.undo_logger_setup()


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




def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    last_step = 0
    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= 100: #Warmup
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
            episode_reward = 0.
            episode += 1
            last_step = step

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    env = NormalizedEnv(gym.make('random1-v0'))

    seed = 42
    validate_steps = 2000
    max_episode_length = 500 * 15
    validate_episodes = 20
    train_iter = 200000

    if seed > 0:
        np.random.seed(seed)
        env.seed(seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, seed)
    evaluate = Evaluator(validate_episodes,
        validate_steps, "./cheetah_gym/agents/weights", max_episode_length=max_episode_length)

    train(train_iter, agent, env, evaluate,
        validate_steps, "./cheetah_gym/agents/weights", max_episode_length=max_episode_length, debug=True)

    # test(validate_episodes, agent, env, evaluate, "./cheetah_gym/agents/weights",
    #     visualize=False, debug=True)
