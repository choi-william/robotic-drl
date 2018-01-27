import sys, math
import numpy as np

import Box2D
from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape, b2RevoluteJointDef)

import gym
from gym import spaces
from gym.utils import seeding

# Double Joint environment for testing deep reinforcement learning algorithms.
# 
# Since we rely on the box2D to simulate object motion, it's particularly important to choose an
# accurate action space for training.

FPS    = 50

# Exterior Box Dimensions
BOX_HEIGHT = 30
BOX_WIDTH = 30

EXT_BOX_POLY = [
    (0, BOX_HEIGHT),
    (0, 0),
    (BOX_WIDTH, 0),
    (BOX_WIDTH, BOX_HEIGHT)
    ]

# Linkage Dimensions
LINK_LENGTH = 20
MOTOR_TORQUE = 80
MOTOR_SPEED = 4 # rad/s (?)

# Tip Start Position
TIP_POS = (BOX_WIDTH/4,BOX_HEIGHT/2)

# Desired object position
TARGET_POS = (10, 10)

# Rendering Parameters
VIEWPORT_W = 500
VIEWPORT_H = 500
SCALE  = VIEWPORT_W/BOX_HEIGHT

class DoubleJoint(gym.Env):
    # rgb_array currently not supported
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None # to be used later for rendering

        # Gravity set to 0; assuming that the plane is flat
        self.world = b2World(gravity=[0,0], doSleep=True)
        self.exterior_box = None
        self.object = None
        self.first_link = None
        self.second_link = None
        self.actuator_mount = None
        self.tip = None

        # Observation Space 
        # [object posx, object posy, joint1 angle, joint2 angle, joint1 angular speed, joint2 angular speed]
        # Note: potentially add 'ball in touch with end effector'
        high = np.array([np.inf]*6)
        self.observation_space = spaces.Box(low=-high,high=high)

        # Continuous action space; one for each joint
        # action space represents angular velocity
        self.action_space = spaces.Box(-1,1,(2,))

        self.prev_shaping = None # for reward calculation
        self.drawlist = [] # for rendering

        self._reset()

    @staticmethod
    def angle_normalize(x):
        # between -1 and 1
        return (((x+np.pi) % (2*np.pi)) - np.pi)/np.pi

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.exterior_box: return
        self.world.DestroyBody(self.exterior_box)
        self.exterior_box = None
        self.world.DestroyBody(self.object)
        self.object = None
        self.world.DestroyBody(self.first_link)
        self.first_link = None
        self.world.DestroyBody(self.second_link)
        self.second_link = None
        self.world.DestroyBody(self.actuator_mount)
        self.actuator_mount = None
        self.world.DestroyBody(self.tip)
        self.tip = None

    def _reset(self):
        self._destroy()
        self.game_over = False # To be used later

        # Creating the Exterior Box that defines the 2D Plane
        exterior_box_fixture = b2FixtureDef(
            shape = b2LoopShape(vertices=EXT_BOX_POLY),
            categoryBits = 0x0001,
            maskBits = 0x0004 | 0x0001 | 0x0002
            )
        self.exterior_box = self.world.CreateStaticBody(position = (0, 0))
        self.exterior_box.CreateFixture(exterior_box_fixture)
        self.exterior_box.color1 = (0,0,1)
        self.exterior_box.color2 = (0,0,0)

        # Creating the object to manipulate
        object_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/40),
            density = 1,
            categoryBits = 0x0002,
            maskBits = 0x0004 | 0x0002 | 0x0001
            )

        self.object = self.world.CreateDynamicBody(
            position = (BOX_WIDTH/2,BOX_HEIGHT/2),
            fixtures = object_fixture,
            linearDamping = 0.3
            )
        self.object.at_target = False
        self.object.at_target_count = 0
        self.object.color1 = (1,1,0)
        self.object.color2 = (0,0,0)

        # Creating the actuator mount
        self.actuator_mount = self.world.CreateStaticBody(
            position = (-BOX_WIDTH/50,BOX_HEIGHT/2), 
            shapes = b2PolygonShape(box=(BOX_WIDTH/50, BOX_WIDTH/30))
            )
        self.actuator_mount.color1 = (1,0,0)
        self.actuator_mount.color2 = (0,0,0)

        # Link Position Calculation
        adj = (BOX_WIDTH/50 + TIP_POS[0])/4
        hyp = (LINK_LENGTH/2)
        opp = np.sqrt(hyp*hyp - adj*adj)

        # Creating the actuator
        first_link_fixture = b2FixtureDef(
            shape = b2PolygonShape(box=(LINK_LENGTH/2,BOX_WIDTH/70,(0,0),math.tan(opp/adj))),
            density = 1, 
            categoryBits = 0x0003,
            maskBits = 0
            )

        second_link_fixture = b2FixtureDef(
            shape = b2PolygonShape(box=(LINK_LENGTH/2,BOX_WIDTH/70,(0,0),math.pi-math.tan(opp/adj))),
            density = 1, 
            categoryBits = 0x0003,
            maskBits = 0
            )

        tip_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/50),
            density = 1,
            categoryBits = 0x0004,
            maskBits = 0x0004 | 0x0002 | 0x0001
            )

        self.first_link = self.world.CreateDynamicBody(
            position = self.actuator_mount.position + (adj,opp),
            fixtures = first_link_fixture
            )
        self.first_link.color1 = (0,0,0)
        self.first_link.color2 = (0,0,0)

        self.first_link.joint = self.world.CreateRevoluteJoint(
            bodyA = self.actuator_mount,
            bodyB = self.first_link,
            anchor = self.actuator_mount.position,
            collideConnected=False,
            maxMotorTorque = MOTOR_TORQUE,
            motorSpeed = 0, 
            enableMotor = True
            )

        self.second_link = self.world.CreateDynamicBody(
            position = self.first_link.position + (adj*2,0),
            fixtures = second_link_fixture
            )
        self.second_link.color1 = (0,0,0)
        self.second_link.color2 = (0,0,0)

        self.second_link.joint = self.world.CreateRevoluteJoint(
            bodyA = self.first_link,
            bodyB = self.second_link,
            anchor = self.first_link.position + (adj,opp),
            collideConnected=False,
            maxMotorTorque = MOTOR_TORQUE,
            motorSpeed = 0,
            enableMotor = True
            )

        self.tip = self.world.CreateDynamicBody(
            position = self.second_link.position + (adj,-opp),
            fixtures = tip_fixture
            )
        self.tip.color1 = (0,1,1)
        self.tip.color2 = (0,0,0)

        self.tip.joint = self.world.CreateWeldJoint(
            bodyA = self.tip,
            bodyB = self.second_link,
            anchor = self.tip.position
            )

        # Adding bodies to be rendered to the drawlist
        self.drawlist = [self.first_link, self.second_link, self.object, self.tip]

        return self._step(np.array([0,0]))[0] # action: zero motor speed

    def _step(self, action):
        # Set motor speeds
        self.first_link.joint.motorSpeed = float(MOTOR_SPEED * np.clip(action[0], -1, 1))
        self.second_link.joint.motorSpeed = float(MOTOR_SPEED * np.clip(action[1], -1, 1))

        # Move forward one frame
        self.world.Step(1.0/FPS, 6*30, 2*30)

        # Observation space (state)
        state = [
            (self.object.position.x - BOX_WIDTH/2)/(BOX_WIDTH/2),
            (self.object.position.y - BOX_HEIGHT/2)/(BOX_HEIGHT/2),
            self.angle_normalize(self.first_link.joint.angle),
            self.angle_normalize(self.second_link.joint.angle),
            self.first_link.joint.speed / MOTOR_SPEED, # -1 to 1 range
            self.second_link.joint.speed / MOTOR_SPEED # -1 to 1 range
            ]
        assert len(state)==6

        # Check if the objects at target position
        dist_to_target = np.sqrt(np.square(TARGET_POS[0]-self.object.position.x) + np.square(TARGET_POS[1]-self.object.position.y)) 
        dist_to_object = np.sqrt(np.square(self.object.position.x-self.tip.position.x) + np.square(self.object.position.y-self.tip.position.y))
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
        reward = -10 * dist_to_target - 10 * dist_to_object
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Reduce reward for using the motor
        for a in action:
            reward -= 0.05 * np.clip(np.abs(a), 0, 1)

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
    env = DoubleJoint()
    s = env.reset()
