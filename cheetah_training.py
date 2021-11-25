import gym
import gym.spaces
import cheetah_gym
import numpy as np
import random

env = gym.make('random1-v0')
# state_space = gym.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), shape=(12,1) ,dtype=np.float32)
env.reset()
while (True):
    env.step(env.action_space.sample())
    pass

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

print(q_table)

num_episodes = 1000
max_steps_per_episode = 10 # but it won't go higher than 1

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01

exploration_decay_rate = 0.01 #if we decrease it, will learn slower

rewards_all_episodes = []

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration -exploitation trade-off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s,a)
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        rewards_current_episode += reward

        if done :
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per 10 episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 100)
count = 100
print("********** Average  reward per thousand episodes **********\n")

for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 100)))
    count += 100

# Print updated Q-table
print("\n\n********** Q-table **********\n")
print(q_table)
