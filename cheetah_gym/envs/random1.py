import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt

from cheetah_gym.walking_simulation import WalkingSimulation
from scipy.spatial.transform import Rotation as R
from collections import deque


class Random1Env(gym.Env):
    MIN_MAX_TORQUE = np.deg2rad(90)
    NUM_JOINTS = 12

    def __init__(self, visualize=False):
        self.sim = WalkingSimulation(visualize=visualize)
        self.action_space = gym.spaces.Box(low=-self.MIN_MAX_TORQUE,
                                           high=self.MIN_MAX_TORQUE,
                                           shape=(self.NUM_JOINTS, ),
                                           dtype=np.float32)
        high = np.ones([38*10], dtype=np.float32)
        # high = np.concatenate([high, np.ones([3], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.initial_position = None
        self.ts = None
        self.goal = None
        self.speed = None
        self.states = deque([np.zeros((38,)) for _ in range(10)], maxlen=10)
        self.last_position = None

    def step(self, action):
        # action = np.insert(action,0,0)
        # action = np.insert(action,3,0)
        # action = np.insert(action,6,0)
        # action = np.insert(action,9,0)

        self.sim.simulation_step(action)
        self.ts += 1

        #TODO : joint angles, joint velocities, root orientation with respect to the world vertical
        #axes, as well as sensory readings including accelerometer, velocimeter, and gyroscope.
        imu_data, leg_data, base_position, contact_points = self.sim.get_state()
        leg_return = np.asarray(leg_data["state"])
        # leg_return = np.delete(leg_return, [0, 3, 6, 9, 13, 16, 19 ,22])
        imu_return = np.asarray(imu_data)
        base_return = np.asarray(base_position)
        contacts = np.asarray(contact_points)
        speed_return = np.asarray([self.speed])
        state = np.concatenate([
                speed_return,
                leg_return,
                base_return,
                imu_return
        ])  # 1, 12, 12, 3, 10

        # Reward function        
        # euler_rot = R.from_quat(imu_data[3:7]).as_euler('zyx', degrees=True)
        time_alive = 10 * self.ts/(240.0) # Timesteps alive
        # angle_to_goal = (np.arctan2(self.goal[1] - base_position[1], self.goal[0] - base_position[0]) - euler_rot[0])
        # f5 = (angle_to_goal * np.asarray(imu_data[7]))
        distance = np.linalg.norm(base_return[0:2] - np.asarray(self.goal[0:2])) # distance
        speed_curr = (base_return[0:2] - self.last_position[0:2]) / 240.0 # speed
        cosine = np.dot(speed_curr, self.speed) / (np.linalg.norm(speed_curr) * np.linalg.norm(self.speed))
        speed_diff = speed_curr
        speed_diff[0] = speed_curr[0] - self.speed
        cosine_reward = np.linalg.norm(speed_diff) * 100 * cosine / 240
        self.last_position = base_return
        self.goal[0] += self.speed / 240.0
        # rotation = -2 * (np.sum(np.abs(euler_rot))) / (240*15) # deviation in orientation
        origin_distance = np.linalg.norm(base_return[0:2] - np.asarray(self.initial_position)[0:2]) / 240
        
        reward =  1 / distance
        # reward = time_alive - (distance/240.0)
        # print("Distance reward: ", f5)
        # print("Euler deviation reward: ", f2)
        # print("Time alive reward: ", f4)

        # Done condition
        base_leg_links = [3,7,11,15]
        # print(contacts)
        contacts = np.delete(contacts, base_leg_links)
        # print(contacts)
        if np.any(contacts):
            reward = -1000
        if (distance > 15):
            reward -= 300
        done = np.any(contacts) or (distance > 15) or np.linalg.norm(base_return[0:2] - np.asarray(self.initial_position[0:2])) > 75

        info = {}

        self.states.append(state)
        concat_state = np.stack(list(self.states), axis=0).reshape(380,)

        return concat_state, reward, done, info

    def reset(self):
        self.sim.reset_robot()
        imu_data, leg_data, base_position, contact_points = self.sim.get_state()
        self.goal = np.asarray(base_position)
        # self.goal[0] += 2.5
        self.speed = max(np.random.random_sample(), 0.5) * 5 * 3.6 # m/s
        self.last_position = np.asarray(base_position)
        # self.speed = 0
        # self.goal = np.random.rand(3,)
        # self.goal[2] = self.initial_z + (np.random.rand() * 0.3)
        # self.goal = self.goal/np.linalg.norm(self.goal)
        
        leg_return = np.asarray(leg_data["state"])
        # leg_return = np.delete(leg_return, [0, 3, 6, 9, 13, 16, 19 ,22])
        imu_return = np.asarray(imu_data)
        base_return = np.asarray(base_position)
        contacts = np.asarray(contact_points)
        speed_return = np.asarray([self.speed])
        state = np.concatenate([
                speed_return,
                leg_return,
                base_return,
                imu_return
        ])  # 1, 12, 12, 3, 10
     
        self.ts = 0
        self.initial_position = base_position
        self.states = deque([np.zeros((38,)) for _ in range(10)], maxlen=10)
        self.states.append(state)
        concat_state = np.stack(list(self.states), axis=0).reshape(380,)

        return concat_state

    def render(self):
        image = self.sim.render()
        # plt.imshow(image)
        # plt.show()
        return image

    def close(self):
        pass
