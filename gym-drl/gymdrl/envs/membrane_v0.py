import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import math

from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

# Membrane environment for testing deep reinforcement learning algorithms.
#
# Failure Conditions and the Reward metric needs to be defined


FPS = 50

# Exterior Box Dimension
BOX_HEIGHT = 30
BOX_WIDTH = 30
EXT_BOX_POLY = [
    (BOX_WIDTH/15, BOX_HEIGHT),
    (BOX_WIDTH/15, 0),
    (BOX_WIDTH-BOX_WIDTH/15, 0),
    (BOX_WIDTH-BOX_WIDTH/15, BOX_HEIGHT)
    ]

NUM_ACTUATORS = 5

# Motor Parameters
MOTOR_SPEED = 10	# m/s
MOTOR_TORQUE = 80

GRAVITY = -30

# Desired Object Position
TARGET_POS = [25,20]

# Rendering Parameters
VIEWPORT_W = 500
VIEWPORT_H = 500
SCALE = VIEWPORT_H/BOX_HEIGHT

ACTUATOR_TRANSLATION_MAX = BOX_HEIGHT/2
ACTUATOR_TRANSLATION_MEAN = ACTUATOR_TRANSLATION_MAX/2
ACTUATOR_TRANSLATION_AMP = ACTUATOR_TRANSLATION_MAX/2

class Membrane(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering

        # Gravity set to 0; assuming that the plane is flat	
        self.world = b2World(gravity=[0,GRAVITY], doSleep=True)
        self.exterior_box = None
        # Five linear actuators 
        self.actuator_list = []
        self.actuator_joint_list = []
        # Linkages
        self.link_left_list = [] # four links
        self.link_right_list = [] # four links
        # Object to be manipulated
        self.object = None 

        # Drawlist for rendering
        self.drawlist = []

        # Observation Space 
        # [object posx, object posy, actuator1 pos.y, ... , actuator5 pos.y, actuator1 speed.y, ... , actuator5 speed.y]
        high = np.array([np.inf]*12)
        self.observation_space = spaces.Box(low=-high,high=high)

        # Continuous action space; one for each linear actuator (5 toal)
        # action space represents velocity
        self.action_space = spaces.Box(-1,1,(5,))
        self.prev_shaping = None # for reward calculation

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.exterior_box: return # return if the exterior box hasn't been created
        self.world.DestroyBody(self.exterior_box)
        self.exterior_box = None
        self.world.DestroyBody(self.object)
        self.object = None

        for actuator in self.actuator_list:
            self.world.DestroyBody(actuator)
        self.actuators = []

        for left_link in self.link_left_list:
            self.world.DestroyBody(left_link)
        self.link_left_list = []

        for right_link in self.link_right_list:
            self.world.DestroyBody(right_link)
        self.link_right_list = []

    def _reset(self):
        self._destroy()
        self.game_over = False # To be used later

		# Creating the Exterior Box that defines the 2D Plane
        self.exterior_box = self.world.CreateStaticBody(
            position = (0, 0),
            shapes = b2LoopShape(vertices=EXT_BOX_POLY)
            )
        self.exterior_box.color1 = (0,0,0)
        self.exterior_box.color2 = (0,0,0)

        # Creating the object to manipulate
        object_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/20),
            density = 1,
            friction = 0.6
            )

        self.object = self.world.CreateDynamicBody(
            position = (5, BOX_HEIGHT/2),
            fixtures = object_fixture
            )
        self.object.at_target = False
        self.object.at_target_count = 0
        self.object.color1 = (1,1,0)
        self.object.color2 = (0,0,0)

        # Creating 5 actuators 
        self.actuator_list = []
        self.actuator_joint_list = []

        actuator_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/50.0),
            density = 1,
            friction = 0.6,
            groupIndex = -1
            )

        for i in range(5):
            actuator = self.world.CreateDynamicBody(
                position = (BOX_WIDTH/5*i+BOX_WIDTH/10, BOX_HEIGHT/10), 
                fixtures = actuator_fixture
                )
            actuator.color1 = (1,0,0)
            actuator.color2 = (1,0,0)

            #SHOULD STAY COMMENTED
            # actuator = self.world.CreateKinematicBody(
            #     position = (BOX_WIDTH/5*i+BOX_WIDTH/10, BOX_HEIGHT/10), 
            #     shapes = b2CircleShape(radius=BOX_WIDTH/50.0), # diameter is 1/5th of the space allocated for each actuator
            #     )

            actuator_joint = self.world.CreatePrismaticJoint(
                bodyA = self.exterior_box,
                bodyB = actuator,
                anchor = actuator.position,
                axis = (0,1),
                lowerTranslation = 0,
                upperTranslation = ACTUATOR_TRANSLATION_MAX,
                enableLimit = True,
                maxMotorForce = 1000.0,
                motorSpeed = 0,
                enableMotor = True
                )
            
            self.actuator_list.append(actuator)
            self.actuator_joint_list.append(actuator_joint)

        # Creating the linkages that will form the semi-flexible membrane
        self.link_left_list = [] # four links
        self.link_right_list = [] # four links

        link_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(BOX_WIDTH/12, BOX_WIDTH/70.0)),
            density=1, 
            friction = 0.6,
            groupIndex = -1 # neg index to prevent collision
            )

        for i in range(1,5):
            link_left = self.world.CreateDynamicBody(
                position = (BOX_WIDTH/5*i-BOX_WIDTH/10+BOX_WIDTH/12,BOX_HEIGHT/10),
                fixtures = link_fixture
                )
            link_left.color1 = (0,1,1)
            link_left.color2 = (1,0,1)
            self.link_left_list.append(link_left)

            link_right = self.world.CreateDynamicBody(
                position = (BOX_WIDTH/5*i+BOX_WIDTH/10-BOX_WIDTH/12,BOX_HEIGHT/10),
                fixtures = link_fixture
                )
            link_right.color1 = (0,1,1)
            link_right.color2 = (1,0,1)
            self.link_right_list.append(link_right)
            
            joint_left = self.world.CreateRevoluteJoint(
                bodyA = self.actuator_list[i-1],
                bodyB = link_left,
                anchor = self.actuator_list[i-1].worldCenter,
                collideConnected=False
                )

            joint_right = self.world.CreateRevoluteJoint(
                bodyA = self.actuator_list[i],
                bodyB = link_right,
                anchor = self.actuator_list[i].worldCenter,
                collideConnected=False
                )

            joint_middle = self.world.CreatePrismaticJoint(
                bodyA = link_left,
                bodyB = link_right,
                anchor = (link_right.position.x-BOX_WIDTH/12+BOX_WIDTH/70, link_right.position.y),
                axis = (1,0),
                lowerTranslation = 0,
                upperTranslation = BOX_WIDTH/6-BOX_WIDTH/35,
                enableLimit = True
                )

        self.drawlist = self.link_right_list + self.link_left_list + [self.object] 
        self.drawlist = self.drawlist + self.actuator_list
        return self._step(np.array([0,0,0,0,0]))[0] # action: zero motor speed

    def _step(self, action):
        # Set motor speeds
        for i, actuator_joint in enumerate(self.actuator_joint_list):
            actuator_joint.motorSpeed = float(MOTOR_SPEED * np.clip(action[i], -1, 1))

        # Move forward one frame
        self.world.Step(1.0/FPS, 6*30, 2*30)

        # Observation space (state)
        state = [
            (self.object.position.x-BOX_WIDTH/2)/(BOX_WIDTH/2),
            (self.object.position.y-BOX_HEIGHT/2)/(BOX_HEIGHT/2),
            (self.actuator_list[0].position.y-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (self.actuator_list[1].position.y-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (self.actuator_list[2].position.y-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (self.actuator_list[3].position.y-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (self.actuator_list[4].position.y-ACTUATOR_TRANSLATION_MEAN)/ACTUATOR_TRANSLATION_AMP,
            (self.actuator_list[0].linearVelocity.y)/MOTOR_SPEED,
            (self.actuator_list[1].linearVelocity.y)/MOTOR_SPEED,
            (self.actuator_list[2].linearVelocity.y)/MOTOR_SPEED,
            (self.actuator_list[3].linearVelocity.y)/MOTOR_SPEED,
            (self.actuator_list[4].linearVelocity.y)/MOTOR_SPEED,
        ]
        assert len(state)==12            

        # Check if the objects at target position
        # dist_to_target = np.sqrt(np.square(TARGET_POS[0]-self.object.position.x) + np.square(TARGET_POS[1]-self.object.position.y)) 
        dist_to_target = TARGET_POS[0]-self.object.position.x
        shaping = 0

        if dist_to_target < 2:
            self.object.at_target = True
            self.object.at_target_count += 1
            shaping += 20
        else:            
            self.object.at_target = False
            self.object.at_target_count = 0

        # Rewards
        # Eventually add a penalty for object velocity at target
        reward = -10 * dist_to_target
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Reduce reward for using the motor
        for a in action:
            reward -= 50 * np.clip(np.abs(a), 0, 1)

        done = False

        # If object is at the target position the task is complete
        if self.object.at_target_count >= 3:
            done = True
            reward += 100

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
            self.viewer.set_bounds(-5, BOX_WIDTH+5, -5, BOX_HEIGHT+5)

        self.viewer.draw_polyline( [(TARGET_POS[0], 0), (TARGET_POS[0], BOX_HEIGHT)], color=(1,0,0) )

        # Target Position Visualized
        self.viewer.draw_polyline( [(TARGET_POS[0], 0), (TARGET_POS[0], BOX_HEIGHT)], color=(1,0,0) )
        self.viewer.draw_polyline( [(0, TARGET_POS[1]), (BOX_WIDTH, TARGET_POS[1])], color=(1,0,0) )

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

if __name__=="__main__":
    env = Membrane()
    s = env.reset()

