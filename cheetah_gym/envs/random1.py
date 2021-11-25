import gym

from cheetah_gym.walking_simulation import WalkingSimulation


class Random1Env(gym.Env):

    def __init__(self):
        self.sim = WalkingSimulation()

    def step(self, action):
        self.sim.simulation_step(action)

        state = self.sim.get_state()
        reward = 1
        done = True
        info = {}

        return state, reward, done, info

    def reset(self):
        self.sim.reset_robot()
        state = 0
        return state

    def render(self, mode):
        pass

    def close(self):
        pass
