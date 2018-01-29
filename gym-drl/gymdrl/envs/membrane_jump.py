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

# MEMBRANE JUMP ENVIRONMENT
#   - bounces the ball at the center of the platform
# 
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

FPS = 50
# Desired Object Position
TARGET_POS = [membrane_base.BOX_WIDTH/2,membrane_base.BOX_HEIGHT/2]

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

class MembraneJump(gym.Env):
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

        self.count = 0
        self.prev_state = None

        # Observation Space 
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
            shape = b2CircleShape(radius=membrane_base.OBJ_SIZE/2),
            density = 0.3,
            friction = 0.6,
            restitution = 0.5
            )
        object_position = (self.np_random.uniform(membrane_base.OBJ_POS_OFFSET,membrane_base.BOX_WIDTH-membrane_base.OBJ_POS_OFFSET), membrane_base.BOX_HEIGHT/3)
        self.object = self.world.CreateDynamicBody(
            position = object_position,
            fixtures = object_fixture,
            linearDamping = 0.3 # Control this parameter for surface friction
            )
        self.object.color1 = (1,1,0)
        self.object.color2 = (0,0,0)

        self.drawlist = self.drawlist + [self.object]

        return self._step(np.array([0,0,0,0,0]))[0] # action: zero motor speed

    def _step(self, action):
        self.count = self.count + 1
        
#        if self.prev_state is not None:
#            action = self.programmed_policy(self.prev_state)

        # Set motor speeds
        for i, actuator in enumerate(self.actuator_list):
            actuator.joint.motorSpeed = float(membrane_base.MOTOR_SPEED * np.clip(action[i], -1, 1))

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
            object_vel[0]/((membrane_base.BOX_WIDTH/16)*FPS),
            object_vel[1]/((membrane_base.BOX_HEIGHT/16)*FPS),
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
        self.prev_state = state

        assert len(state)==14            

        # Rewards
        reward = 0
        shaping = -200*np.abs(TARGET_POS[1]-object_pos[1])/membrane_base.BOX_HEIGHT - 200*np.abs(TARGET_POS[0]-object_pos[0])/membrane_base.BOX_WIDTH - 10*np.abs(state[2]) + 300*(object_pos[1] - max(actuator_pos))/TARGET_POS[1]

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

        membrane_base.render_helper(self)

        # Target Position Visualized
        self.viewer.draw_polyline( [(TARGET_POS[0], 0), (TARGET_POS[0], membrane_base.BOX_HEIGHT)], color=(1,0,0) )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def programmed_policy(self, state):

        FAST_SPEED = 1;
        HIGH_SPEED = 0.8;
        MEDIUM_SPEED = 0.5;
        SLOW_SPEED = 0.1;

        ACTUATOR_START = membrane_base.BOX_SIDE_OFFSET
        ACTUATOR_SPACING = membrane_base.GAP

        act_pos = [(membrane_base.BOX_SIDE_OFFSET+membrane_base.GAP*i) for i in range(5)]

        p = (self.object.position.x-membrane_base.BOX_SIDE_OFFSET)/(membrane_base.GAP)
        action = -FAST_SPEED*np.ones(5)

        if (TARGET_POS[0]-self.object.position.x) > 0:
            #move right
            action[int(np.ceil(p))] = MEDIUM_SPEED + 1.5*np.sin(0.1*self.count)
            action[int(np.floor(p))] = HIGH_SPEED + 1.5*np.sin(0.1*self.count)
        else:
            #move left
            action[int(np.floor(p))] = MEDIUM_SPEED + 1.5*np.sin(0.1*self.count)
            action[int(np.ceil(p))] = HIGH_SPEED + 1.5*np.sin(0.1*self.count)
        return action

if __name__=="__main__":
    env = MembraneJump()
    s = env.reset()
