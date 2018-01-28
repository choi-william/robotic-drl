import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape)

import numpy as np
import math

# The following is to give import access for membrane_base
import gymdrl
import sys
sys.path.append(gymdrl.__file__[:-11] + 'envs') #hacky but necessary
import membrane_base

# MEMBRANE ROTATE ENVIRONMENT
#   - rotates a square object by a specified angle: 90, 180, 270, 360, etc
# 
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

FPS = 50
# Set the rotation angle for the task here
ROTATION_ANGLE = np.pi
# Set the object size for the square here
OBJ_SIZE = membrane_base.BOX_WIDTH*0.2

########################
# Rendering Parameters #
########################
TEMPW = membrane_base.BOX_WIDTH
TEMPH = membrane_base.BOX_HEIGHT_BELOW_ACTUATORS + membrane_base.BOX_HEIGHT
VIEWPORT_W = int(1000*TEMPW / (TEMPW + TEMPH))
VIEWPORT_H = int(1000*TEMPH / (TEMPW + TEMPH))

####################
# Noise Parameters #
####################
OBJ_POS_STDDEV = membrane_base.BOX_WIDTH/100.0
OBJ_VEL_STDDEV = 0 # Nothing set currently
ACTUATOR_POS_STDDEV = membrane_base.BOX_WIDTH/100.0
ACTUATOR_VEL_STDDEV = 0 # Nothing set currently

class MembraneRotate(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering
        # Object to be manipulated
        self.object = None 
        # Initializing other common components in the environment
        membrane_base.init_helper(self)

        self.prev_state = None

        # Observation Space 
        high = np.array([np.inf]*15)
        self.observation_space = spaces.Box(low=-high,high=high)

        # Continuous action space; one for each linear actuator (5 total)
        # action space represents velocity
        self.action_space = spaces.Box(-1,1,(5,))
        self.prev_shaping = None # for reward calculation

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.exterior_box: return # return if the exterior box hasn't been created
        membrane_base.destroy_helper(self)
        self.world.DestroyBody(self.object)
        self.object = None

    def _reset(self):
        self._destroy()
        membrane_base.reset_helper(self)

        # Creating the object to manipulate
        object_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(OBJ_SIZE/2, OBJ_SIZE/2)),
            density = 1,
            friction = 0.6,
            restitution = 0.0
            )
        
        object_position = (membrane_base.BOX_WIDTH/2, OBJ_SIZE/2 + membrane_base.LINK_HEIGHT)
        self.object = self.world.CreateDynamicBody(
            position = object_position,
            fixtures = object_fixture,
            linearDamping = 0.6 # Control this parameter for surface friction
            )
        self.object.color1 = (1,1,0)
        self.object.color2 = (0,0,0)

        self.drawlist = self.drawlist + [self.object]

        return self._step(np.array([0,0,0,0,0]))[0] # action: zero motor speed

    def _step(self, action):
        # Set motor speeds
        for i, actuator in enumerate(self.actuator_list):
            actuator.joint.motorSpeed = float(MOTOR_SPEED * np.clip(action[i], -1, 1))

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
        object_angle = angle_normalize(self.object.angle)
        
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
            object_vel[0]/((membrane_base.BOX_WIDTH/16)*FPS),
            object_vel[1]/((membrane_base.BOX_HEIGHT/16)*FPS),
            object_angle/180.0,
            (actuator_pos[0]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[1]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[2]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[3]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[4]-membrane_base.ACTUATOR_TRANSLATION_MEAN)/membrane_base.ACTUATOR_TRANSLATION_AMP,
            (actuator_vel[0])/membrane_base.MOTOR_SPEED,
            (actuator_vel[1])/membrane_base.MOTOR_SPEED,
            (actuator_vel[2])/membrane_base.MOTOR_SPEED,
            (actuator_vel[3])/membrane_base.MOTOR_SPEED,
            (actuator_vel[4])/membrane_base.MOTOR_SPEED,
        ]
        assert len(state)==15            

        # Rewards
        reward = 0

        shaping = -200*np.abs(state[4]) - 100*np.abs(state[2]) - 100*np.abs(state[3])

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Reduce reward for using the motor
        for a in action:
            reward -= 0.05*np.clip(np.abs(a), 0, 1)

        done = False

        if np.abs(object_angle) < 5.0 and np.abs(state[2]) < 0.0001 and np.abs(state[3]) < 0.0001:
            reward += 100
            done = True

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

        membrane_base.render_helper(self)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x-ROTATION_ANGLE+np.pi) % (2*np.pi)) - np.pi)*180/np.pi

if __name__=="__main__":
    env = MembraneRotate()
    s = env.reset()

