"""
this script uses the imu data from the real robot instead of quaternion value to train the tensegrity robot in simulation
"""

import os
import copy
import os.path
import time
from typing import Any, Optional, SupportsFloat
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import csv

# from rospkg import RosPack
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv

from EMAfilter import EMAFilter
from tensegrity_utils import *


INITIALIZE_ROBOT_IN_AIR = False
TEST_STEP_RATE = 0.5
INITIAL_TENSION = 0.0
CONTROL_UPDATE_FREQ = 20 # Hz
LOG_TO_CSV = False
# target = ["com_pos", "com_vel", "action", "imu_data", "tendon_length", "tendon_speed", "tension_force", "velocity_reward", "ang_momentum_reward", "ang_momentum_penalty", "tension_penalty", "contorl_penalty", "reward"]
LOG_FILE = '/logs/PPO7/angular_momentum_penalty.csv'
LOG_TARGET = 'ang_momentum_penalty'


class TensegrityEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100}
    info_keywords = ("rew_forward_x", "rew_ang_vel_pitch", "penalty_ang_vel_pitch")
    # TODO
    # 1.add action normalization
    # 2.add curriculum_learning
    # 3.use velocity cmd(angular)
    # 4.separate observation calculation function
    # 5.separate reward calculation function
    # 6.separate linear and angular momentum calculation functions
    # 7.add camera following
    # 8.consider terminate situation
    # 9.consider curriculum assistive force?
    # 10.add plot for debug in test mode

    def __init__(self, act_range, test, max_steps, resume=False, **kwargs):
        """
        resume training is abandoned due to mujoco supports evaluation along with training
        """
            
        # ema filter
        self.ema_filter = EMAFilter(0.267, np.array([0.0]*36))
        # initial tendon length
        self.prev_ten_length = None

        self.test = test
        self.is_params_set = False
        self.prev_action = None

        self.max_step = max_steps  # max_steps of one sub-env, used for curriculum learning
        self.resume = resume
        self.act_range = act_range  # tension force range
        print("act_range: ", self.act_range)

        self.max_episode = 512 * 2  # maximum steps of every episode

        self.initial_tension = INITIAL_TENSION
        
        # control range
        self.num_actions = 24
        self.ctrl_max = np.array([0.] * self.num_actions)
        self.ctrl_min = np.array([-self.act_range] * self.num_actions)
        self.action_space_low = [-1.0] * self.num_actions
        self.action_space_high = [1.0] * self.num_actions
        self.control_update_freq = CONTROL_UPDATE_FREQ
        self.control_count_max = 100 / self.control_update_freq
        self.control_count = 0


        # observation parameters
        self.projected_gravity = np.zeros(18)  # (3*6,)
        self.linear_velocity = None # (3*6,)
        self.com_pos = np.zeros(3)  # (3,)    
        self.com_velocity = None    # (6,)
        self.angular_velocity = None    # (3*6,)
        self.imu_data = None    # (6*6,)
        self.contact_state = None   # (2*6,): 1 for contact, 0 for no contact
        self.distance_from_com = None   # (3*6,)
        self.tendon_length = None   # (24,)
        self.tendon_speed = None    # (24,)
        self.actions = np.array([0.]*self.num_actions)  # (24,)
        self.vel_command = np.array([0., 0., 0.])   # (3,)
        self.tension_force = np.array([0.]*self.num_actions)  # (24,)

        # observation space
        num_obs_per_step_actor = 111
        num_obs_per_step_critic = 204
        actor_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs_per_step_actor,))
        critic_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs_per_step_critic,))
        observation_space = spaces.Dict({
            "actor": actor_observation_space, 
            "critic": critic_observation_space
            })

        self.check_steps = 200
        self.com_pos_deque = deque(maxlen=self.check_steps)
        for k in range(self.check_steps):
            self.com_pos_deque.appendleft(np.zeros(3))


        self.episode_cnt = 0  # current episode step counter, used in test mode, reset to zero at the beginning of new episode
        self.step_cnt = 0  # never reset

        self.step_rate = 0.
        if self.test:
            self.step_rate = TEST_STEP_RATE

        #self.rospack = RosPack()
        root_path = os.path.dirname(os.path.abspath(__file__)) + "/.."
        self.log_to_csv = LOG_TO_CSV
        if self.log_to_csv:
            self.log_file = root_path + LOG_FILE

        model_path = root_path + '/models/scene_real_model.xml'
        self.frame_skip = 2  # number of mujoco simulation steps per action step
        MujocoEnv.__init__(
            self,
            model_path,
            self.frame_skip,  # frame_skip
            observation_space=observation_space,
            **kwargs)

        utils.EzPickle.__init__(self)

        # robot initial state
        self.init_robot_in_air = INITIALIZE_ROBOT_IN_AIR # flag for reset the robot in air
        self.default_init_qpos = np.array([-0.125,  0.,  0.25,  1.,  0., 0.,  0.,
                                           0.125,  0.,  0.25, 1.,  0.,  0.,  0.,
                                           0., 0.125,  0.25,  0.70710678,  0.,  0.70710678, 0.,
                                           0., -0.125,  0.25,  0.70710678, 0.,  0.70710678,  0.,
                                           0.,  0., 0.375,  0.70710678,  0.70710678,  0.,  0.,
                                           0.,  0.,  0.125,  0.70710678,  0.70710678, 0.,  0.])
        self.default_init_qvel = np.array([0.0]*36)
        self.body_inertial = [np.diag(self.model.body_inertia[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                                                "link{}".format(i+1))]) for i in range(6)]

        self.prev_com_pos = np.array([0., 0., 0.])
        self.reset_model()

    def _set_render_param(self):
        if self.test:
            self.mujoco_renderer.viewer._render_every_frame = False

    def _set_action_space(self):
        """
        always use normalized action space
        Noting: during env.step(), please rescale the action to the actual range!
        """
        low = np.asarray(self.action_space_low)
        high = np.asarray(self.action_space_high)
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        return self.action_space

    def _get_current_actor_obs(self):
        """
        actor_obs = imu_data + tendon_length + tendon_speed + actions + commands
        actor observation dimention == 36 + 24 + 24 + 24 + 3 = 111
        """
        self.imu_data = self.ema_filter.update(np.array(copy.deepcopy(self.data.sensordata[:36]))) # (36,)
        imu_with_noise = self.imu_data * np.random.uniform(1.0 - self.step_rate * 0.05, 1.0 + self.step_rate * 0.05, 36)
        self.tendon_length = np.array(copy.deepcopy(self.data.ten_length)) * np.random.uniform(1.0 - self.step_rate * 0.05, 1.0 + self.step_rate * 0.05, 24)  # (24,)
        self.tendon_speed = np.array(copy.deepcopy(self.data.ten_velocity)) * np.random.uniform(1.0 - self.step_rate * 0.05, 1.0 + self.step_rate * 0.05, 24) # (24,)

        return np.concatenate((imu_with_noise, self.tendon_length, self.tendon_speed, self.actions, self.vel_command), dtype=np.float32)
    
    def _get_current_critic_obs(self):
        """
        critic_obs = projected gravity + linear velocity + com_velocity + com_pos + angular velocity + imu_data + contact state + distance from COM
            + tendon_length + tendon_speed + actions + commands
        critic observation dimention == 18 + 18 + 3 + 6 + 18 + 36 + 12 + 18 + 24 + 24 + 24 + 3 = 204
        """
        for i in range(6):
            rot_mat = self.data.xmat[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link{}".format(i+1))].reshape(3, 3)
            self.projected_gravity[i*3:i*3+3] = rot_mat.T @ np.array([0., 0., -1.0])
        cur_velocity = np.array(copy.deepcopy(self.data.qvel.reshape(-1, 6)))   # (6, 6)
        self.linear_velocity = np.array(cur_velocity[:, 0:3]).reshape(-1)  # (18,)
        self.com_velocity = np.mean(cur_velocity, axis=0)  # (6,)
        self.com_pos = np.mean(copy.deepcopy(self.data.qpos.reshape(-1, 7)[:, 0:3]), axis=0)  # (3,)
        self.angular_velocity = np.array(cur_velocity[:, 3:]).reshape(-1)  # (18,)
        self.imu_data = self.ema_filter.update(np.array(copy.deepcopy(self.data.sensordata[:36])))  # (36,)
        self.contact_state = np.array(copy.deepcopy(self.data.sensordata[36:48]))  # (12,)
        #print("contact_state", self.contact_state)
        self.distance_from_com = (np.array(copy.deepcopy(self.data.qpos.reshape(-1, 7)[:, 0:3])) - self.com_pos).flatten()  # (18,)
        self.tendon_length = np.array(copy.deepcopy(self.data.ten_length))  # (24,)
        self.tendon_speed = np.array(copy.deepcopy(self.data.ten_velocity))  # (24,)

        return np.concatenate((self.projected_gravity, self.linear_velocity, self.com_velocity, self.com_pos, self.angular_velocity, self.imu_data, 
                               self.contact_state, self.distance_from_com, 
                               self.tendon_length, self.tendon_speed, self.actions, self.vel_command), dtype=np.float32)
    
    def _get_current_obs(self):
        """
        observations = Dict({"actor":actor_obs, "critic":critic_obs})
        """
        actor_obs = self._get_current_actor_obs()
        critic_obs = self._get_current_critic_obs()
        return {"actor": actor_obs, "critic": critic_obs}

    
    """  def save_log_data(self, step, log_data):
        with open(self.log_file, 'a') as f:
            writer = csv.writer(f)
            data = [step] + list(log_data)
            writer.writerow(data) """
        
    def step(self, action):
        """
        what we need do inside the step():
            - rescale_actions and add assistive force if needed---> TODO
            - filter action value
            - mujoco simulation step forward
            - update flag and counters(such as step_cnt)
            - calculate the observations
            - calculate reward
            - check terminate conditions and truncated condition separately(timeout): reference->https://github.com/openai/gym/issues/2510
            - return
        action: (24,) normalized actions[-1,1] directly from policy
        """
        if not self.is_params_set:
            self._set_render_param()
            self.is_params_set = True
        
        if self.prev_action is None:
            self.prev_action = action

        if self.prev_ten_length is None:
            self.prev_ten_length = np.array(self.data.ten_length)

        # add external disturbance to center of each rod--> [N]
        self.data.qfrc_applied[:] = 0.02 * self.step_rate * np.random.randn(len(self.data.qfrc_applied))

        # add external assistive force curriculum
        self.data.xfrc_applied[:] = 0.0

        # update control input
        if self.control_count > self.control_count_max: # self.control_update_count_max=5
            # rescale action to tension force first
            self.actions = action 
            self.tension_force = rescale_actions(self.ctrl_min, self.ctrl_max, action)   
            # add action(tension force) noise from [0.95, 1.05]--> percentage
            self.tension_force *= np.random.uniform(1.0 - self.step_rate * 0.05, 1.0 + self.step_rate * 0.05, self.num_actions)
            self.control_count = 0
 
        # update control count
        self.control_count += 1

        # do simulation
        self._step_mujoco_simulation(self.tension_force, self.frame_skip)  # self.frame_skip=2, mujoco_step=200hz [0.95, 1.05]

        # update flags
        self.episode_cnt += 1
        self.step_cnt += 1

        # calculate the observations and update the observation deque
        cur_step_obs = self._get_current_obs()

        self.com_pos = np.mean(copy.deepcopy(self.data.qpos.reshape(-1, 7)[:, 0:3]), axis=0)  # (3,)
        self.com_vel = np.mean(copy.deepcopy(self.data.qvel.reshape(-1, 6)), axis=0)  # (6,)

        obs = cur_step_obs

        # calculate the rewards
        current_ang_momentum = calculate_angular_momentum(self.data,
                                                                self.model,
                                                               self.data.qpos,
                                                               self.data.qvel,
                                                               self.com_pos,
                                                               self.com_vel[0:3],
                                                                self.body_inertial)

        self.prev_com_pos = self.com_pos

        if np.dot(self.com_vel[0:2], self.vel_command[0:2]) > np.linalg.norm(self.vel_command)**2: # if v_x * v_x_cmd > ||v_x_cmd||^2
            self.velocity_reward = 1.0
        else:
            # velocity_reward = e^(-12*(v_x - v_x_cmd)^2)
            self.velocity_reward = np.exp(-10.0*(np.dot(self.com_vel[0:2], self.vel_command[0:2]) - np.linalg.norm(self.vel_command[0:2])**2)**2)
            #self.velocity_reward = np.exp(-10.0*(np.dot(current_com_vel[0:2], self.vel_command[0:2]) - np.linalg.norm(self.vel_command[0:2]))**2)
            #self.velocity_reward = np.exp(-8.0*np.square(current_com_vel[0:2] - self.vel_command[0:2]).sum())
        self.ang_momentum_penalty = current_ang_momentum[1] * int(current_ang_momentum[1] < 0.)
        self.ang_momentum_reward = current_ang_momentum[1] * int(current_ang_momentum[1] > 0.)
        
        #self.action_penalty = -0.000 * np.linalg.norm(action) * self.step_rate # pre 0.001
        self.tension_penalty = -0.0050 * np.linalg.norm(self.tension_force) * self.step_rate
        self.contorl_penalty = -0.000 * np.linalg.norm(action - self.prev_action) * self.step_rate
        #self.current_step_total_reward = self.velocity_reward + 1.5 * self.ang_momentum_reward + 5.0 * self.ang_momentum_penalty + self.action_penalty + self.contorl_penalty
        self.current_step_total_reward = self.velocity_reward + 1.5 * self.ang_momentum_reward + 5.0 * self.ang_momentum_penalty + self.tension_penalty + self.contorl_penalty

        # log data to csv
        if self.test and self.log_to_csv:
            if LOG_TARGET == 'com_pos':
                save_log_data(self.step_cnt, self.com_pos, self.log_file)
            elif LOG_TARGET == 'com_vel':
                save_log_data(self.step_cnt, self.com_vel, self.log_file)
            elif LOG_TARGET == 'action':
                save_log_data(self.step_cnt, rescale_actions(self.ctrl_min, self.ctrl_max, action), self.log_file)
            elif LOG_TARGET == 'imu_data':
                save_log_data(self.step_cnt, self.data.sensordata[0:36], self.log_file)
            elif LOG_TARGET == 'tendon_length':
                save_log_data(self.step_cnt, self.data.ten_length, self.log_file)
            elif LOG_TARGET == 'tendon_speed':
                save_log_data(self.step_cnt, self.data.ten_velocity, self.log_file)
            elif LOG_TARGET == 'tension_force':
                save_log_data(self.step_cnt, self.tension_force, self.log_file)
            elif LOG_TARGET == 'velocity_reward':
                save_log_data(self.step_cnt, np.array([self.velocity_reward]), self.log_file)
            elif LOG_TARGET == 'ang_momentum_reward':
                save_log_data(self.step_cnt, np.array([self.ang_momentum_reward]), self.log_file)
            elif LOG_TARGET == 'ang_momentum_penalty':
                save_log_data(self.step_cnt, np.array([self.ang_momentum_penalty]), self.log_file)
            elif LOG_TARGET == 'tension_penalty':
                save_log_data(self.step_cnt, np.array([self.tension_penalty]), self.log_file)
            elif LOG_TARGET == 'contorl_penalty':
                save_log_data(self.step_cnt, np.array([self.contorl_penalty]), self.log_file)
            elif LOG_TARGET == 'reward':
                save_log_data(self.step_cnt, np.array([self.current_step_total_reward]), self.log_file)
        
        ## update prev_action
        self.prev_action = action
        # update prev_ten_length
        self.prev_ten_length = np.array(self.data.ten_length)

        rew_dict = {
            "ang_momentum_reward": self.ang_momentum_reward,
            "angular_momentum_penalty": self.ang_momentum_penalty
        }

        # check terminate and truncated
        self.com_pos_deque.appendleft(self.com_pos)
        terminated = False
        if self.episode_cnt > 400:
            terminated = np.linalg.norm(self.com_pos_deque[0] - self.com_pos_deque[-1]) < 0.03
        if terminated:
            self.current_step_total_reward += -5.0

        truncated = not (self.episode_cnt < self.max_episode)

        # nan check
        if np.any(np.isnan(obs["actor"])):
            print("NaN in actor obs")
            raise ValueError
        if np.any(np.isnan(obs["critic"])):
            print("NaN in critic obs")
            raise ValueError
        if np.any(np.isnan(self.current_step_total_reward)):
            print("NaN in reward calculation")
            raise ValueError
        
        # print("current obs: ", obs)

        return (
            obs,
            self.current_step_total_reward,
            terminated,
            truncated,
            rew_dict
        )

    def reset_model(self):

        self.episode_cnt = 0

        # update step_rate and max_episode value at the beginning of every episode
        if self.max_step is not None:  # training or resume training mode
            self.step_rate = min(float(self.step_cnt) / self.max_step, 1)
        # self.max_episode = 512 + 1024 * self.step_rate

        # sample random initial pose
        qpos_addition = np.random.uniform(-0.05, 0.05, len(self.default_init_qpos)) * self.step_rate  # TODO:BUG

        qpos = self.default_init_qpos + qpos_addition
        if (self.init_robot_in_air and self.step_rate > 0.2) or self.test:
            qpos += np.array([0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0,
                              0, 0, 1, 0, 0, 0, 0
                              ]) * np.random.uniform(0.00, 1.00)

        # sample random initial vel
        """ qvel_addition = np.random.uniform(-0.1, 0.1, len(self.default_init_qvel)) * self.step_rate
        qvel = self.default_init_qvel + qvel_addition """
        qvel = self.default_init_qvel

        self.set_state(qpos, qvel)  # reset the values of mujoco model(robot)

        # switch to new command
        if self.test:
            self.vel_command = [0.6, 0.0, 0.0]
        else:
            #v = np.random.uniform(0.6, 0.6+0.2*self.step_rate)
            v = 0.6
            self.vel_command = [v, 0.0, 0.0]

        # initialize ema filter
        self.ema_filter = EMAFilter(0.267, np.array([0.0]*36))
        
        # initial tendon length
        self.prev_ten_length = self.data.ten_length

        # update the com state
        self.prev_com_pos = np.mean(copy.deepcopy(self.data.qpos.reshape(-1, 7)[:, 0:3]), axis=0)  # (3,)
        for k in range(self.check_steps):
            self.com_pos_deque.appendleft(self.prev_com_pos)
        
        # add initial tension
        self.data.ctrl[:] = self.initial_tension

        # return the stacked obs as the initial obs of episode
        return self._get_current_obs()