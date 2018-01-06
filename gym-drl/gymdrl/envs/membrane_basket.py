import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import math

from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

# MEMBRANE BOUNCE ENVIRONMENT
# 
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

FPS = 50
# Desired Object Position
TARGET_POS = [5,15]
BASKET_WIDTH = 8
GRAVITY = -30

##########################
# Exterior Box Dimension #
##########################
BOX_WIDTH = 30
BOX_HEIGHT = 30
BOX_HEIGHT_BELOW_ACTUATORS = 5
EXT_BOX_POLY = [
    (0, BOX_HEIGHT),
    (0, -BOX_HEIGHT_BELOW_ACTUATORS),
    (BOX_WIDTH, -BOX_HEIGHT_BELOW_ACTUATORS),
    (BOX_WIDTH, BOX_HEIGHT)
    ]

###################
# Body Dimensions #
###################
OBJ_SIZE = 0.15 # fraction of box width
OBJ_POS_OFFSET = 0.1 # fraction of box width; should be greater than half the object size
ACTUATOR_TIP_SIZE = 0.05 # fraction of box width
# Distance between the wall and the center of the first actuator
BOX_SIDE_OFFSET = 0.03 # fraction of box width
LINK_WIDTH = 0.2 # fraction of box width
LINK_HEIGHT = 0.04 # fraction of box width
# Do not modify
GAP = (1-BOX_SIDE_OFFSET*2)/4

####################
# Motor Parameters #
####################
MOTOR_SPEED = 25    # m/s
MOTOR_TORQUE = 80

########################
# Rendering Parameters #
########################
VIEWPORT_W = 500
VIEWPORT_H = 500

ACTUATOR_TRANSLATION_MAX = BOX_HEIGHT/3
ACTUATOR_TRANSLATION_MEAN = ACTUATOR_TRANSLATION_MAX/2
ACTUATOR_TRANSLATION_AMP = ACTUATOR_TRANSLATION_MAX/2

####################
# Noise Parameters #
####################
OBJ_POS_STDDEV = BOX_WIDTH/100.0
OBJ_VEL_STDDEV = 0 # Nothing set currently
ACTUATOR_POS_STDDEV = BOX_WIDTH/100.0
ACTUATOR_VEL_STDDEV = 0 # Nothing set currently


#################################
# Reward Calculation Parameters #
#################################
MAX_DIST_TO_TARGET = np.sqrt(np.square(BOX_WIDTH) + np.square(BOX_HEIGHT))
# Maximum distance adjacent actuators can be apart veritically due to the membrane
MAX_VERT_DIST_BETWEEN_ACTUATORS = BOX_WIDTH/4
# Maximum steps at the target before the episode is deemed to be successfully completed
MAX_TARGET_COUNT = 100

class MembraneBasket(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    # Flag that indicates whether to run the env with or without linkages
    with_linkage = True

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering

        self.world = b2World(gravity=[0,GRAVITY], doSleep=True)
        self.exterior_box = None
        
        self.basketL = None
        self.basketR = None

        # Five linear actuators 
        self.actuator_list = []
        # Object to be manipulated
        self.object = None 
        # Linkages
        if self.with_linkage:
            self.link_left_list = [] # four links
            self.link_right_list = [] # four links

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
        self.world.DestroyBody(self.exterior_box)
        self.exterior_box = None
        self.world.DestroyBody(self.object)
        self.object = None

        for actuator in self.actuator_list:
            self.world.DestroyBody(actuator)
        self.actuator_list = []

        if self.with_linkage:
            for left_link in self.link_left_list:
                self.world.DestroyBody(left_link)
            self.link_left_list = []

            for right_link in self.link_right_list:
                self.world.DestroyBody(right_link)
            self.link_right_list = []

    def _reset(self):
        self._destroy()

        # Creating the baskets
        self.basketL = self.world.CreateStaticBody(
            position = (TARGET_POS[0]-BASKET_WIDTH/2, TARGET_POS[1]),
            shapes = b2CircleShape(radius=BOX_WIDTH*ACTUATOR_TIP_SIZE/4)
            )
        self.basketR = self.world.CreateStaticBody(
            position = (TARGET_POS[0]+BASKET_WIDTH/2, TARGET_POS[1]),
            shapes = b2CircleShape(radius=BOX_WIDTH*ACTUATOR_TIP_SIZE/4)
            )
        self.basketL.color1 = (0,0,0)
        self.basketL.color2 = (0,0,0)
        self.basketR.color1 = (0,0,0)
        self.basketR.color2 = (0,0,0)       
           
        # Creating the Exterior Box that defines the 2D Plane
        self.exterior_box = self.world.CreateStaticBody(
            position = (0, 0),
            shapes = b2LoopShape(vertices=EXT_BOX_POLY)
            )
        self.exterior_box.color1 = (0,0,0)
        self.exterior_box.color2 = (0,0,0)

        # Creating the object to manipulate
        object_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH*OBJ_SIZE/2),
            density = 0.3,
            friction = 0.6,
            restitution = 0.0
            )
        # Randomizing object's initial position
        # object_position = (
        #     self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_WIDTH-BOX_WIDTH*OBJ_POS_OFFSET),
        #     BOX_HEIGHT/5
        #     )
        # object_position = (
        #     self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_WIDTH-BOX_WIDTH*OBJ_POS_OFFSET),
        #     self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_HEIGHT-BOX_WIDTH*OBJ_POS_OFFSET)
        #     )
        object_position = (self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_WIDTH-BOX_WIDTH*OBJ_POS_OFFSET), BOX_HEIGHT/5)
        self.object = self.world.CreateDynamicBody(
            position = object_position,
            fixtures = object_fixture,
            linearDamping = 0.3 # Control this parameter for surface friction
            )
        self.object.at_target = False
        self.object.at_target_count = 0
        self.object.color1 = (1,1,0)
        self.object.color2 = (0,0,0)

        # Creating 5 actuators 
        actuator_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH*ACTUATOR_TIP_SIZE/2),
            density = 1,
            friction = 0.6,
            restitution = 0.0,
            groupIndex = -1
            )

        for i in range(5):
            actuator = self.world.CreateDynamicBody(
                position = ((BOX_SIDE_OFFSET+GAP*i)*BOX_WIDTH, 0), 
                fixtures = actuator_fixture
                )
            actuator.color1 = (0,0,0.5)
            actuator.color2 = (0,0,0)

            actuator.joint = self.world.CreatePrismaticJoint(
                bodyA = self.exterior_box,
                bodyB = actuator,
                anchor = actuator.position,
                axis = (0,1),
                lowerTranslation = 0,
                upperTranslation = ACTUATOR_TRANSLATION_MAX,
                enableLimit = True,
                maxMotorForce = 100000.0,
                motorSpeed = 0,
                enableMotor = True
                )
            
            self.actuator_list.append(actuator)

        self.drawlist = self.actuator_list + [self.object,self.basketL,self.basketR] 

        if self.with_linkage:
            # Creating the linkages that will form the semi-flexible membrane
            link_fixture = b2FixtureDef(
                shape=b2PolygonShape(box=(LINK_WIDTH*BOX_WIDTH/2, LINK_HEIGHT*BOX_WIDTH/2)),
                density=1, 
                friction = 0.6,
                restitution = 0.0,
                groupIndex = -1 # neg index to prevent collision
                )

            for i in range(4):
                link_left = self.world.CreateDynamicBody(
                    position = (BOX_WIDTH*(BOX_SIDE_OFFSET+GAP*i+LINK_WIDTH/2),0),
                    fixtures = link_fixture
                    )
                link_left.color1 = (0,1,1)
                link_left.color2 = (1,0,1)
                self.link_left_list.append(link_left)

                link_right = self.world.CreateDynamicBody(
                    position = (BOX_WIDTH*(BOX_SIDE_OFFSET+GAP*(i+1)-LINK_WIDTH/2),0),
                    fixtures = link_fixture
                    )
                link_right.color1 = (0,1,1)
                link_right.color2 = (1,0,1)
                self.link_right_list.append(link_right)
                
                joint_left = self.world.CreateRevoluteJoint(
                    bodyA = self.actuator_list[i],
                    bodyB = link_left,
                    anchor = self.actuator_list[i].worldCenter,
                    collideConnected=False
                    )

                joint_right = self.world.CreateRevoluteJoint(
                    bodyA = self.actuator_list[i+1],
                    bodyB = link_right,
                    anchor = self.actuator_list[i+1].worldCenter,
                    collideConnected=False
                    )

                joint_middle = self.world.CreatePrismaticJoint(
                    bodyA = link_left,
                    bodyB = link_right,
                    anchor = (link_right.position.x-BOX_WIDTH*(LINK_WIDTH/2+LINK_HEIGHT/2), link_right.position.y),
                    axis = (1,0),
                    lowerTranslation = 0,
                    upperTranslation = BOX_WIDTH*LINK_WIDTH*2/3,
                    enableLimit = True
                    )
            # Adding linkages to the drawlist
            self.drawlist = self.link_left_list + self.link_right_list + self.drawlist

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
            (object_pos[0]-BOX_WIDTH/2)/(BOX_WIDTH/2),
            (object_pos[1]-BOX_HEIGHT/2)/(BOX_HEIGHT/2),
            object_vel[0]/((BOX_WIDTH/16)*FPS),
            object_vel[1]/((BOX_HEIGHT/16)*FPS),
            (actuator_pos[0]-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[1]-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[2]-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[3]-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[4]-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (actuator_vel[0])/MOTOR_SPEED,
            (actuator_vel[1])/MOTOR_SPEED,
            (actuator_vel[2])/MOTOR_SPEED,
            (actuator_vel[3])/MOTOR_SPEED,
            (actuator_vel[4])/MOTOR_SPEED,
        ]
        assert len(state)==14            

        # Rewards
        reward = 0
        shaping = -200*np.abs(TARGET_POS[1]-object_pos[1])/BOX_HEIGHT - 200*np.abs(TARGET_POS[0]-object_pos[0])/BOX_WIDTH - 10*np.abs(state[2]) + 300*(object_pos[1] - max(actuator_pos))/TARGET_POS[1]

        if (object_pos[1] - max(actuator_pos)) > 4:
            shaping += 20
        
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Reduce reward for using the motor
        for a in action:
            reward -= 0.05*np.clip(np.abs(a), 0, 1)

        if np.abs(TARGET_POS[0]-object_pos[0]) < BASKET_WIDTH/2 and np.abs(TARGET_POS[1]-object_pos[1]) < 2: 
            if object_vel[1] < 0:
                reward += 200
            else:
                reward -= 300
                
        done = False
        
        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(-5, BOX_WIDTH+5, -5-BOX_HEIGHT_BELOW_ACTUATORS, BOX_HEIGHT+5)

        # Actuator start position visualized
        self.viewer.draw_polyline( [(0, 0), (BOX_WIDTH, 0)], color=(1,0,1) )

#        # Target Position Visualized
#        self.viewer.draw_polyline( [(TARGET_POS[0], 0), (TARGET_POS[0], BOX_HEIGHT)], color=(1,0,0) )
#        self.viewer.draw_polyline( [(0, TARGET_POS[1]), (BOX_WIDTH, TARGET_POS[1])], color=(1,0,0) )

        # Exterior Box Visualized
        box_fixture = self.exterior_box.fixtures[0]
        box_trans = box_fixture.body.transform
        box_path = [box_trans*v for v in box_fixture.shape.vertices]
        box_path.append(box_path[0])
        self.viewer.draw_polyline(box_path, color=self.exterior_box.color2, linewidth=2)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class MembraneWithoutLinkages(MembraneBasket):
    with_linkage = False

if __name__=="__main__":
    env = MembraneBasket()
    s = env.reset()
