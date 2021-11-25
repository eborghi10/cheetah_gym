import gym
import gym.spaces
import numpy as np

from cheetah_gym.walking_simulation import WalkingSimulation


class Random1Env(gym.Env):
    MIN_MAX_TORQUE = 5.0
    NUM_JOINTS = 12

    def __init__(self):
        self.sim = WalkingSimulation()
        self.action_space = gym.spaces.Box(low=-self.MIN_MAX_TORQUE,
                                           high=self.MIN_MAX_TORQUE,
                                           shape=(self.NUM_JOINTS, ),
                                           dtype=np.float32)
        high = np.inf * np.ones([37], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.initial_z = None
        self.last_x = None
        self.last_y = None
        self.ts = None

    def step(self, action):
        self.sim.simulation_step(action)
        self.ts += 1

        imu_data, leg_data, base_position, contact_points = self.sim.get_state()
        state = np.concatenate([
                np.asarray(imu_data),
                np.asarray(leg_data["state"]),
                np.asarray(base_position)
        ])  # 10, 12+12, 3

        # Reward function
        f1 = imu_data[7]
        f2 = -3 * ((self.last_y - base_position[1])**2)
        f3 = -50 * ((base_position[2] - self.initial_z) / self.initial_z)**2
        f4 = 25 * self.ts/500.0
        f5 = 0.  # -0.02 * np.sum(u_i**2)
        reward = f1 + f2 + f3 + f4 + f5

        # Done condition
        done = base_position[2] < self.initial_z/2.0

        info = {}

        return state, reward, done, info

    def reset(self):
        self.sim.reset_robot()
        imu_data, leg_data, base_position, contact_points = self.sim.get_state()
        state = np.concatenate([
                np.asarray(imu_data),
                np.asarray(leg_data["state"]),
                np.asarray(base_position)
        ])
        self.last_x = base_position[0]
        self.last_y = base_position[1]
        self.initial_z = base_position[2]
        self.ts = 0
        return state

    def render(self, mode):
        return self.sim.render()

    def close(self):
        pass
