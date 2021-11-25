import gym
import numpy as np

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        norm = abs(self.action_space.high[0] -  self.action_space.low[0])
        return action / norm

    def reverse_action(self, action):
        norm = abs(self.action_space.high[0] -  self.action_space.low[0])
        return action * norm
