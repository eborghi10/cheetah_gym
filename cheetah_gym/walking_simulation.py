#!/usr/bin/env python
import numpy as np
import os
import pybullet as p
import pybullet_data
import random

from pybullet_utils import gazebo_world_parser


class WalkingSimulation(object):
    def __init__(self, visualize=False):
        self.terrain = "random1"
        self.get_last_vel = [0] * 3
        self.robot_height = 0.40 #0.31
        self.motor_id_list = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.init_new_pos = [0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6, 0.0, -0.8, 1.6,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.lateralFriction = 1.0
        self.spinningFriction = 0.0065
        self.stand_kp = 100.0
        self.stand_kd = 1.0
        self.joint_kp = 10.0
        self.freq = 500.0  # Hz
        self.joint_kd = 0.2

        self.__init_simulator(visualize=visualize)

    def __init_simulator(self, visualize=False):
        robot_start_pos = [0, 0, self.robot_height]
        # p.connect(p.GUI if visualize else p.DIRECT)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setAdditionalSearchPath(os.path.dirname(os.path.realpath(__file__)))
        p.resetSimulation()
        # p.setTimeStep(1.0 / 60.) # Docs state that is recommended not to change it. Default is 240
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(0.2, 45, -30, [1, -1, 1])

        heightPerturbationRange = 0.06
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
        # for j in range(int(numHeightfieldColumns/2)):
        #     for i in range(int(numHeightfieldRows/2)):
        #         height = random.uniform(0, heightPerturbationRange)
        #         heightfieldData[2*i+2*j*numHeightfieldRows] = height
        #         heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
        #         heightfieldData[2*i+(2*j+1)*numHeightfieldRows] = height
        #         heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height
        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[.05, .05, 1],
            heightfieldTextureScaling=(numHeightfieldRows-1)/2,
            heightfieldData=heightfieldData,
            numHeightfieldRows=numHeightfieldRows,
            numHeightfieldColumns=numHeightfieldColumns)
        self.ground_id = p.createMultiBody(0, terrainShape)
        p.resetBasePositionAndOrientation(self.ground_id, [0, 0, 0], [0, 0, 0, 1])
        p.changeDynamics(self.ground_id, -1, lateralFriction=self.lateralFriction)

        # Disable visualization of cameras in pybullet GUI
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # Enable this if you want better performance
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.boxId = p.loadURDF(r"cheetah_gym\urdf\mini_cheetah.urdf", robot_start_pos, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)
        p.changeDynamics(self.boxId, 3, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.boxId, 7, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.boxId, 11, spinningFriction=self.spinningFriction)
        p.changeDynamics(self.boxId, 15, spinningFriction=self.spinningFriction)

        self.reset_robot()

    def reset_robot(self):
        robot_z = self.robot_height
        p.resetBasePositionAndOrientation(
            self.boxId, [0, 0, robot_z], [0, 0, 0, 1])
        p.resetBaseVelocity(self.boxId, [0, 0, 0], [0, 0, 0])
        for j in range(12):
            p.resetJointState(
                self.boxId, self.motor_id_list[j], self.init_new_pos[j], self.init_new_pos[j+12])

        p.setJointMotorControlArray(bodyUniqueId=self.boxId,
                                    jointIndices=self.motor_id_list,
                                    controlMode=p.POSITION_CONTROL,
                                    forces=[12.]*len(self.motor_id_list),
                                    targetPositions=np.zeros((12,)))

    def simulation_step(self, tau):
        # set tau to simulator
        p.setJointMotorControlArray(bodyUniqueId=self.boxId,
                                    jointIndices=self.motor_id_list,
                                    controlMode=p.POSITION_CONTROL,
                                    forces=[12.]*len(self.motor_id_list),
                                    targetPositions=tau)
        # import pdb; pdb.set_trace()

        p.stepSimulation()

    def __get_motor_joint_states(self, robot):
        joint_number_range = range(p.getNumJoints(robot))
        joint_states = p.getJointStates(robot, joint_number_range)
        joint_infos = [p.getJointInfo(robot, i) for i in joint_number_range]
        joint_states, joint_name = zip(*[(j, i[1], ) for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED])
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques, joint_name

    def get_state(self):
        get_matrix = []
        get_velocity = []
        get_invert = []
        imu_data = [0] * 10
        leg_data = {}
        leg_data["state"] = [0] * 24
        leg_data["name"] = [""] * 12

        base_pose = p.getBasePositionAndOrientation(self.boxId)

        get_velocity = p.getBaseVelocity(self.boxId)
        get_invert = p.invertTransform(base_pose[0], base_pose[1])
        get_matrix = p.getMatrixFromQuaternion(get_invert[1])

        # IMU data
        imu_data[3] = base_pose[1][0]
        imu_data[4] = base_pose[1][1]
        imu_data[5] = base_pose[1][2]
        imu_data[6] = base_pose[1][3]

        imu_data[7] = get_matrix[0] * get_velocity[1][0] + get_matrix[1] * \
            get_velocity[1][1] + get_matrix[2] * get_velocity[1][2]
        imu_data[8] = get_matrix[3] * get_velocity[1][0] + get_matrix[4] * \
            get_velocity[1][1] + get_matrix[5] * get_velocity[1][2]
        imu_data[9] = get_matrix[6] * get_velocity[1][0] + get_matrix[7] * \
            get_velocity[1][1] + get_matrix[8] * get_velocity[1][2]

        # calculate the acceleration of the robot
        linear_X = (get_velocity[0][0] - self.get_last_vel[0]) * self.freq
        linear_Y = (get_velocity[0][1] - self.get_last_vel[1]) * self.freq
        linear_Z = 9.8 + (get_velocity[0][2] - self.get_last_vel[2]) * self.freq
        imu_data[0] = get_matrix[0] * linear_X + \
            get_matrix[1] * linear_Y + get_matrix[2] * linear_Z
        imu_data[1] = get_matrix[3] * linear_X + \
            get_matrix[4] * linear_Y + get_matrix[5] * linear_Z
        imu_data[2] = get_matrix[6] * linear_X + \
            get_matrix[7] * linear_Y + get_matrix[8] * linear_Z

        # joint data
        joint_positions, joint_velocities, _, joint_names = \
            self.__get_motor_joint_states(self.boxId)
        leg_data["state"][0:12] = joint_positions
        leg_data["state"][12:24] = joint_velocities
        leg_data["name"] = joint_names

        # CoM velocity
        self.get_last_vel = [get_velocity[0][0], get_velocity[0][1], get_velocity[0][2]]

        # Contacts
        contact_points = [np.asarray(p.getContactPoints(bodyA=self.boxId, bodyB=self.ground_id, linkIndexA=i)).shape != (0,) for i in range(16)]
        # print("CONTACT", contact_points)
        # print("CONTACTS: ", temp)

        return imu_data, leg_data, base_pose[0], contact_points

    def render(self):
        cam_dist = 3
        cam_yaw = 0
        cam_pitch = -30
        render_width = 320
        render_height = 240

        base_pos = [0, 0, 0]
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(render_width)/render_height,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
            width=render_width,
            height=render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    def __del__(self):
        p.disconnect()
