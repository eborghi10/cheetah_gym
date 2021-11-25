from threading import Lock
import concurrent
import concurrent.futures
from copy import deepcopy
import gym

from cheetah_gym.envs.normalized_env import NormalizedEnv
from cheetah_gym.agents.evaluator import Evaluator
from cheetah_gym.agents.ddpg import DDPG
from cheetah_gym.agents.util import *

global_steps = 0
seed = 42
agent = DDPG(37, 12, seed)
max_episode_length = 50000
lock = Lock()
env = NormalizedEnv(gym.make('random1-v0'))


def train_single():
    global global_steps, env
    print(f"Training on thread, acculumated steps : {global_steps}")
    with lock:
        env_cpy = deepcopy(env)
        agent_cpy = deepcopy(agent)
    agent_cpy.is_training = True
    episode_steps = 0
    steps = 0
    episode_reward = 0.
    observation = None
    done = False
    while not done:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env_cpy.reset())
            agent_cpy.reset(observation)

        # agent pick action ...
        if steps <= 100:  # Warmup
            action = agent_cpy.random_action()
        else:
            action = agent_cpy.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env_cpy.step(action)
        observation2 = deepcopy(observation2)
        if steps >= (max_episode_length - 1):
            done = True

        # agent observe and update policy
        agent_cpy.observe(reward, observation2, done)
        if steps > 100:  # Warmup
            agent_cpy.update_policy()

        # update
        steps += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

    # Save result
    return steps, (
        observation,
        agent_cpy.select_action(observation),
        0.,
        False
    )


def train2(num_iterations, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    global global_steps, env
    agent.is_training = True
    episodes = 0

    NUM_WORKERS = 6
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for _ in range(num_iterations):
            futures.append(executor.submit(train_single))  # .add_done_callback(evaluator_fn)

        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                print(f"Error when querying result, got {e}")
                continue
            with lock:
                agent.memory.append(*res[1])
            with lock:
                global_steps += res[0]
            episodes += 1
            if (episodes % 100) == 0:
                with lock:
                    policy = lambda x: agent.select_action(x, decay_epsilon=False)
                    validate_reward = evaluate(env, policy, debug=False, visualize=False)
                    if debug: prYellow(f'[Evaluate] Step_{global_steps:07d}: mean_reward:{validate_reward}')

                    # [optional] save intermediate model
                    agent.save_model(output)


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
        if step <= 100:  # Warmup
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
        if step > 100:  # Warmup
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

    seed = 42
    validate_steps = 2000
    max_episode_length = 500
    validate_episodes = 20
    train_iter = 200000

    evaluate = Evaluator(validate_episodes,
        validate_steps, "./cheetah_gym/agents/weights", max_episode_length=max_episode_length)

    train2(train_iter, evaluate,
        validate_steps, "./cheetah_gym/agents/weights", max_episode_length=max_episode_length, debug=True)
