import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import math

#the following is to give import access for membrane_base

import gymdrl
import sys
sys.path.append(gymdrl.__file__[:-11] + 'envs') #hacky but necessary

import membrane_base

from gym.envs.classic_control import rendering



# MEMBRANE BOUNCE ENVIRONMENT
# 
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

FPS = 50

####################
# Noise Parameters #
####################
OBJ_POS_STDDEV = membrane_base.BOX_WIDTH/100.0
OBJ_VEL_STDDEV = 0 # Nothing set currently
ACTUATOR_POS_STDDEV = membrane_base.BOX_WIDTH/100.0
ACTUATOR_VEL_STDDEV = 0 # Nothing set currently

#################################
# Reward Calculation Parameters #
#################################
# Desired Object Position
TARGET_POS = [24,9]

MAX_DIST_TO_TARGET = np.sqrt(np.square(membrane_base.BOX_WIDTH) + np.square(membrane_base.BOX_HEIGHT))
# Maximum distance adjacent actuators can be apart veritically due to the membrane
MAX_VERT_DIST_BETWEEN_ACTUATORS = membrane_base.BOX_WIDTH/4
# Maximum steps at the target before the episode is deemed to be successfully completed
MAX_TARGET_COUNT = 100

########################
# Rendering Parameters #
########################
TEMPW = membrane_base.BOX_WIDTH
TEMPH = membrane_base.BOX_HEIGHT_BELOW_ACTUATORS + membrane_base.BOX_HEIGHT

VIEWPORT_W = int(1000*TEMPW / (TEMPW + TEMPH))
VIEWPORT_H = int(1000*TEMPH / (TEMPW + TEMPH))

class MembraneCalibration(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    # Flag that indicates whether to run the env with or without linkages
    with_linkage = True

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering
       
        membrane_base.init_helper(self)

        # Drawlist for rendering
        self.drawlist = []

        # Observation Space 
        # [object posx, object posy, actuator1 pos.y, ... , actuator5 pos.y, actuator1 speed.y, ... , actuator5 speed.y]
        high = np.array([np.inf]*14)
        self.observation_space = spaces.Box(low=-high,high=high)

        # Continuous action space; one for each linear actuator (5 total)
        # action space represents velocity
        self.action_space = spaces.Box(-1,1,(5,))
        self.prev_shaping = None # for reward calculation

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _check_actuator_pos(self):
        result = True
        for i in range(4):
            dist_diff = np.abs(self.actuator_list[i+1].position.y - self.actuator_list[i].position.y)
            if dist_diff > MAX_VERT_DIST_BETWEEN_ACTUATORS:
                result = False
                break
        return result

    def _destroy(self):
        if not self.exterior_box: return # return if the exterior box hasn't been created
        membrane_base.destroy_helper(self)

    def _reset(self):
        self._destroy()

        membrane_base.reset_helper(self)
        
        return self._step(np.array([0,0,0,0,0]))[0] # action: zero motor speed

    def _step(self, action):

        # Set motor speeds
        for i, actuator in enumerate(self.actuator_list):
            actuator.joint.motorSpeed = float(membrane_base.MOTOR_SPEED * np.clip(action[i], -1, 1))
            actuator.joint.motorSpeed = float(membrane_base.MOTOR_SPEED * 0.5)


        # Move forward one frame
        self.world.Step(1.0/FPS, 6*30, 2*30)

        # Required values to be acquired from the platform
        object_pos = [
            np.random.normal(self.object.position.x, OBJ_POS_STDDEV),
            np.random.normal(self.object.position.y, OBJ_POS_STDDEV)
            ]
        object_vel = [
            self.object.linearVelocity.x,
            self.object.linearVelocity.y
            ]
        actuator_pos = [
            np.random.normal(self.actuator_list[0].position.y, ACTUATOR_POS_STDDEV),
            np.random.normal(self.actuator_list[1].position.y, ACTUATOR_POS_STDDEV),
            np.random.normal(self.actuator_list[2].position.y, ACTUATOR_POS_STDDEV),
            np.random.normal(self.actuator_list[3].position.y, ACTUATOR_POS_STDDEV),
            np.random.normal(self.actuator_list[4].position.y, ACTUATOR_POS_STDDEV)
            ]
        actuator_vel = [
            self.actuator_list[0].linearVelocity.y,
            self.actuator_list[1].linearVelocity.y,
            self.actuator_list[2].linearVelocity.y,
            self.actuator_list[3].linearVelocity.y,
            self.actuator_list[4].linearVelocity.y
            ]

        # Observation space (state)
        state = [
            (object_pos[0]-membrane_base.BOX_WIDTH/2)/(membrane_base.BOX_WIDTH/2),
            (object_pos[1]-membrane_base.BOX_HEIGHT/2)/(membrane_base.BOX_HEIGHT/2),
            2*object_vel[0]/membrane_base.MOTOR_SPEED,
            2*object_vel[1]/membrane_base.MOTOR_SPEED,
            (actuator_pos[0]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[1]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[2]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[3]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[4]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            2*(actuator_vel[0])/membrane_base.MOTOR_SPEED,
            2*(actuator_vel[1])/membrane_base.MOTOR_SPEED,
            2*(actuator_vel[2])/membrane_base.MOTOR_SPEED,
            2*(actuator_vel[3])/membrane_base.MOTOR_SPEED,
            2*(actuator_vel[4])/membrane_base.MOTOR_SPEED,
        ]

        print(actuator_vel[0]/membrane_base.MOTOR_SPEED)
        assert len(state)==14            

        # Rewards
        reward = 0
        shaping = -200*np.abs(TARGET_POS[1]-object_pos[1])/membrane_base.BOX_HEIGHT - 100*np.abs(state[2]) + 300*(object_pos[1] - max(actuator_pos))/TARGET_POS[1]

        if (object_pos[1] - max(actuator_pos)) > 4:
            shaping += 20
        
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Reduce reward for using the motor
        for a in action:
            reward -= 0.05*np.clip(np.abs(a), 0, 1)

        if np.abs(TARGET_POS[1]-object_pos[1]) < 1 and object_vel[1] < 0.05:
            reward += 100
        done = False
        
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, membrane_base.BOX_WIDTH, -membrane_base.BOX_HEIGHT_BELOW_ACTUATORS, membrane_base.BOX_HEIGHT)

        # Target Position Visualized
        self.viewer.draw_polyline( [(TARGET_POS[0], 0), (TARGET_POS[0], membrane_base.BOX_HEIGHT)], color=(1,0,0) )
        self.viewer.draw_polyline( [(0, TARGET_POS[1]), (membrane_base.BOX_WIDTH, TARGET_POS[1])], color=(1,0,0) )

        membrane_base.render_helper(self)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__=="__main__":
    env = Membrane()
    s = env.reset()
