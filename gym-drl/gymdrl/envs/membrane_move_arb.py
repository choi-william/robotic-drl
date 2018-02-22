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

from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)


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

class MembraneMoveArb(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    # Flag that indicates whether to run the env with or without linkages
    with_linkage = True

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering
        self.target_pos = None
        membrane_base.init_helper(self)
        
        self.object = None

        # Drawlist for rendering
        self.drawlist = []
        
        self.prev_state = None

        # Observation Space 
        # [object posx, object posy, actuator1 pos.y, ... , actuator5 pos.y, actuator1 speed.y, ... , actuator5 speed.y]
        high = np.array([np.inf]*16)
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
        object_position = (membrane_base.BOX_WIDTH/2, membrane_base.OBJ_SIZE/2 + membrane_base.LINK_HEIGHT)
        self.object = self.world.CreateDynamicBody(
            position = object_position,
            fixtures = object_fixture,
            linearDamping = 0.6 # Control this parameter for surface friction
            )
        self.object.color1 = (1,1,0)
        self.object.color2 = (0,0,0)

        self.drawlist = self.drawlist + [self.object]

        self.target_pos = [np.random.rand()*(membrane_base.BOX_WIDTH-membrane_base.OBJ_SIZE)+membrane_base.OBJ_SIZE/2,np.random.rand()+5]
        
        return self._step(np.array([0,0,0,0,0]))[0] # action: zero motor speed

    def _step(self, action):
        
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
            (self.target_pos[0]-membrane_base.BOX_WIDTH/2)/(membrane_base.BOX_WIDTH/2),
            (self.target_pos[1]-membrane_base.BOX_HEIGHT/2)/(membrane_base.BOX_HEIGHT/2),
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
        self.prev_state = state

        assert len(state)==16            

        # Rewards
        reward = 0
        shaping = \
        -200*np.abs(self.target_pos[0]-object_pos[0])/membrane_base.BOX_WIDTH  \
        -200*np.abs(self.target_pos[1]-object_pos[1])/membrane_base.BOX_HEIGHT  \
        - 50*np.abs(state[2]) \
        - 50*np.abs(state[3])
        
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        if (np.abs(object_pos[0] - self.target_pos[0])) < 1:
            if (np.abs(object_pos[1] - self.target_pos[1])) < 1:
                reward += 50

        # Reduce reward for using the motor
        for a in action:
            reward -= 5*np.clip(np.abs(a), 0, 1)

        done = False
        
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, membrane_base.BOX_WIDTH, -membrane_base.BOX_HEIGHT_BELOW_ACTUATORS, membrane_base.BOX_HEIGHT)

        # Target Position Visualized
        self.viewer.draw_polyline( [(self.target_pos[0], 0), (self.target_pos[0], membrane_base.BOX_HEIGHT)], color=(1,0,0) )
        self.viewer.draw_polyline( [(0, self.target_pos[1]), (membrane_base.BOX_WIDTH, self.target_pos[1])], color=(1,0,0) )

        membrane_base.render_helper(self)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def programmed_policy(self,state):

        FAST_SPEED = 1;
        MEDIUM_SPEED = 0.5;
        SLOW_SPEED = 0.1;
        VERY_SLOW_SPEED = 0.05;

        ACTUATOR_START = membrane_base.BOX_SIDE_OFFSET
        ACTUATOR_SPACING = membrane_base.GAP

        p = (self.object.position.x-ACTUATOR_START)/ACTUATOR_SPACING
        
        q = (self.target_pos[0]-ACTUATOR_START)/ACTUATOR_SPACING

        action = -SLOW_SPEED*np.ones(5)

        if (self.target_pos[0]-self.object.position.x) > 0:
            #move right
            action[int(np.floor(p))] = VERY_SLOW_SPEED
            if int(np.ceil(q)+1) <= 4:
                action[int(np.ceil(q)+1)] = VERY_SLOW_SPEED
        else:
            #move left
            if int(np.ceil(q)-1) >= 0:
                action[int(np.ceil(q)-1)] = VERY_SLOW_SPEED
                
            action[int(np.ceil(p))] = VERY_SLOW_SPEED
            
        return action

if __name__=="__main__":
    env = Membrane()
    s = env.reset()
