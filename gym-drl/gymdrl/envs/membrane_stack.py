import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import math

from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape)

# The following is to give import access for membrane_base
import gymdrl
import sys
sys.path.append(gymdrl.__file__[:-11] + 'envs') #hacky but necessary
import membrane_base

# MEMBRANE STACK ENVIRONMENT
#   - stacks two square objects on top of one another
# 
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

FPS = 50
# Set the object size for the square here
OBJ_SIZE = membrane_base.BOX_WIDTH*0.15 # fraction of box width

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

class MembraneStack(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering
        # Objects to be manipulated
        self.object1 = None 
        self.object2 = None
        # Initializing other common components in the environment
        membrane_base.init_helper(self)

        self.prev_state = None

        # Observation Space 
        # [object posx, object posy, actuator1 pos.y, ... , actuator5 pos.y, actuator1 speed.y, ... , actuator5 speed.y]
        high = np.array([np.inf]*18)
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
        self.world.DestroyBody(self.object1)
        self.world.DestroyBody(self.object2)
        self.object1 = None
        self.object2 = None

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
        
        ### Make initial position of objects random. This is similar to how it would be on the robot
        # object1_position = (OBJ_SIZE/2-membrane_base.BOX_WIDTH*0.06, OBJ_SIZE/2 + membrane_base.LINK_HEIGHT/2) ## Yellow box
        min_x = membrane_base.BOX_WIDTH*0.06
        max_x = membrane_base.BOX_WIDTH-OBJ_SIZE/2
        object1_position = (np.random.uniform()* (max_x - min_x), OBJ_SIZE/2 + membrane_base.LINK_HEIGHT/2) ## Yellow box
        object2_position = (np.random.uniform()* (max_x - min_x), OBJ_SIZE*1.6 + membrane_base.LINK_HEIGHT/2) ## green box
        self.object1 = self.world.CreateDynamicBody(
            position = object1_position,
            fixtures = object_fixture,
            linearDamping = 0.8 # Control this parameter for surface friction
            )
        self.object1.color1 = (1,1,0)
        self.object1.color2 = (0,0,0)
        self.object2 = self.world.CreateDynamicBody(
            position = object2_position,
            fixtures = object_fixture,
            linearDamping = 0.8 # Control this parameter for surface friction
            )
        self.object2.color1 = (0,1,0)
        self.object2.color2 = (0,0,0)
        
        self.drawlist = self.drawlist + [self.object1, self.object2]

        return self._step(np.array([0,0,0,0,0]))[0] # action: zero motor speed

    def _step(self, action):
        # Set motor speeds
        for i, actuator in enumerate(self.actuator_list):
            actuator.joint.motorSpeed = float(membrane_base.MOTOR_SPEED * np.clip(action[i], -1, 1))

        # Move forward one frame
        ### Add some magic simulation sauce...
        substeps = 10
        solver_iterations=10
        for step in range(substeps):
            self.world.Step((1.0/FPS) * (1.0/substeps), 6*solver_iterations, 2*solver_iterations)

        # Required values to be acquired from the platform
        noise_adjust = 0.2
        object1_pos = [
            np.random.normal(self.object1.position.x, OBJ_POS_STDDEV*noise_adjust),
            np.random.normal(self.object1.position.y, OBJ_POS_STDDEV*noise_adjust)
            ]
        object2_pos = [
            np.random.normal(self.object2.position.x, OBJ_POS_STDDEV*noise_adjust),
            np.random.normal(self.object2.position.y, OBJ_POS_STDDEV*noise_adjust)
            ]
        object1_vel = [
            self.object1.linearVelocity.x,
            self.object1.linearVelocity.y
            ]
        object2_vel = [
            self.object2.linearVelocity.x,
            self.object2.linearVelocity.y
            ]
        # print("ACTUATOR_POS_STDDEV: ", ACTUATOR_POS_STDDEV,  " OBJ_POS_STDDEV: ", OBJ_POS_STDDEV*noise_adjust)
        actuator_pos = [
            np.random.normal(self.actuator_list[0].position.y, ACTUATOR_POS_STDDEV*noise_adjust),
            np.random.normal(self.actuator_list[1].position.y, ACTUATOR_POS_STDDEV*noise_adjust),
            np.random.normal(self.actuator_list[2].position.y, ACTUATOR_POS_STDDEV*noise_adjust),
            np.random.normal(self.actuator_list[3].position.y, ACTUATOR_POS_STDDEV*noise_adjust),
            np.random.normal(self.actuator_list[4].position.y, ACTUATOR_POS_STDDEV*noise_adjust)
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
            (object1_pos[0]-membrane_base.BOX_WIDTH/2)/(membrane_base.BOX_WIDTH/2),
            (object1_pos[1]-membrane_base.BOX_HEIGHT/2)/(membrane_base.BOX_HEIGHT/2),
            (object2_pos[0]-membrane_base.BOX_WIDTH/2)/(membrane_base.BOX_WIDTH/2),
            (object2_pos[1]-membrane_base.BOX_HEIGHT/2)/(membrane_base.BOX_HEIGHT/2),
            object1_vel[0]/((membrane_base.BOX_WIDTH/16)*FPS),
            object1_vel[1]/((membrane_base.BOX_HEIGHT/16)*FPS),
            object2_vel[0]/((membrane_base.BOX_WIDTH/16)*FPS),
            object2_vel[1]/((membrane_base.BOX_WIDTH/16)*FPS),
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
        assert len(state)==18            

        # Rewards
        reward = 0

        # distance between the objects
        obj_dist_x = object2_pos[0] - object1_pos[0]
        obj_dist_y = object2_pos[1] - object1_pos[1]
        reward = ((-1*np.abs(object2_pos[0]-object1_pos[0]))  +
                  (-1*np.abs(object2_pos[1]-object1_pos[1]) ))

        shaping = -200*np.abs(obj_dist_y)/membrane_base.BOX_HEIGHT -150*np.abs(obj_dist_x)/membrane_base.BOX_WIDTH
        
        """
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        """
        
        # Reduce reward for using the motor
        for a in action:
            reward -= 0.05*np.clip(np.abs(a), 0, 1)

        done = False

        ## stacked in motion
        if (np.abs(obj_dist_x) < 2.0 and np.abs(obj_dist_y) < 4.2 ## positions
            # and np.abs(state[4]) < 0.01 and np.abs(state[5]) < 0.01 ## velocities 
            # and np.abs(state[6]) < 0.01 and np.abs(state[7]) < 0.01
            ):
            reward += 2 - np.abs(obj_dist_x)
            
        ### Stacked at rest
        if (np.abs(obj_dist_x) < 1.5 and np.abs(obj_dist_y) < 4.2 ## positions
            and np.abs(state[4]) < 0.01 and np.abs(state[5]) < 0.01 ## velocities 
            and np.abs(state[6]) < 0.01 and np.abs(state[7]) < 0.01):
            reward += 10
            done = True

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

        membrane_base.render_helper(self)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__=="__main__":
    env = MembraneStack()
    s = env.reset()

