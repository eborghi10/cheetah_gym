import numpy as np
from threading import Lock
import concurrent
import functools
import argparse
from copy import deepcopy
import gym

from cheetah_gym.envs.normalized_env import NormalizedEnv
from cheetah_gym.agents.evaluator import Evaluator
from cheetah_gym.agents.ddpg import DDPG
from cheetah_gym.agents.util import *


def train_single(agent):
    env = gym.make('random1-v0')
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]
    agent_cpy = deepcopy(agent)
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    done = False
    while not done:
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

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

    return [
        observation,
        agent.select_action(observation),
        0.,
        False]

def train2(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    episodes = episode_steps = 0
    last_step = 0
    observation = None
    train_fn = functools.partial(train_single, agent)
    NUM_WORKERS = 5
    while episodes < num_iterations:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for _ in range(NUM_WORKERS):
                futures.append(executor.submit(train_fn))
        for future in futures:
            agent.memory.append(future.result())
            episodes+=1






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
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow(f'[Evaluate] Step_{step:07d}: mean_reward:{validate_reward}')

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


if __name__ == "__main__":

    env = NormalizedEnv(gym.make('random1-v0'))

    seed = 42
    validate_steps = 2000
    max_episode_length = 500
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


    # train2(train_iter, agent, env, evaluate,
    #     validate_steps, "./cheetah_gym/agents/weights", max_episode_length=max_episode_length, debug=True)
    train(train_iter, agent, env, evaluate,
        validate_steps, "./cheetah_gym/agents/weights", max_episode_length=max_episode_length, debug=True)
