import gym
import gym.spaces
import numpy as np

from cheetah_gym.walking_simulation import WalkingSimulation
from scipy.spatial.transform import Rotation as R


class Random1Env(gym.Env):
    MIN_MAX_TORQUE = np.pi/2
    NUM_JOINTS = 12

    def __init__(self):
        self.sim = WalkingSimulation()
        self.action_space = gym.spaces.Box(low=-self.MIN_MAX_TORQUE,
                                           high=self.MIN_MAX_TORQUE,
                                           shape=(self.NUM_JOINTS, ),
                                           dtype=np.float32)
        # high = np.inf * np.ones([40], dtype=np.float32)
        high = 50.0 * np.ones([37], dtype=np.float32)
        high = np.concatenate([high, np.ones([3], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.initial_z = None
        self.last_x = None
        self.last_y = None
        self.ts = None
        self.last_velocities = None
        self.goal = None

    def step(self, action):
        self.sim.simulation_step(action)
        self.ts += 1

        #TODO : joint angles, joint velocities, root orientation with respect to the world vertical
        #axes, as well as sensory readings including accelerometer, velocimeter, and gyroscope.
        imu_data, leg_data, base_position, contact_points = self.sim.get_state()
        state = np.concatenate([
                np.asarray(imu_data),
                np.asarray(leg_data["state"]),
                np.asarray(base_position),
                self.goal
        ])  # 10, 12+12, 3 , 3

        # Reward function
        # f1 = imu_data[7]
        # declare f1 as the reward element using velocity with respect a goal_vector

        
        euler_rot = R.from_quat(imu_data[3:7]).as_euler('zyx', degrees=True)
        
        f1 = -1 * (imu_data[0] ** 2)
        f2 = -0.5 * (euler_rot[2] ** 2)
        f3 = -5 * ((base_position[2] - self.initial_z))**2 ## deviation in Z
        f4 = 5 * self.ts/500.0
        # angle_to_goal = (np.arctan2(self.goal[1] - base_position[1], self.goal[0] - base_position[0]) - euler_rot[0])
        # f5 = (angle_to_goal * np.asarray(imu_data[7]))
        f5 = 1 * (base_position[0] - self.last_x)
        f6 = -5 * ((base_position[1] - self.last_y)**2)
        # f7 = -0.02 * np.sum(np.asarray(self.last_velocities - leg_data["state"][12:24])**2)
        
        reward = f5

        # Done condition
        done = base_position[2] < self.initial_z * 0.6
        self.last_velocities = leg_data["state"][12:24]
        # self.last_y = base_position[1]

        info = {}

        return state, reward, done, info

    def reset(self):
        self.sim.reset_robot()
        imu_data, leg_data, base_position, contact_points = self.sim.get_state()
        self.initial_z = base_position[2]
        self.goal = np.random.rand(3,)
        self.goal[2] = self.initial_z + (np.random.rand() * 0.3)
        self.goal = self.goal/np.linalg.norm(self.goal)
        
        state = np.concatenate([
                np.asarray(imu_data),
                np.asarray(leg_data["state"]),
                np.asarray(base_position),
                self.goal
        ])
        self.last_x = base_position[0]
        self.last_y = base_position[1]        
        self.ts = 0
        self.last_velocities = leg_data["state"][12:24]

        return state

    def render(self, mode):
        return self.sim.render()

    def close(self):
        pass
