# HARDWARE MEMBRANE ENVIRONMENT
#
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

import gym
import cv2
from gym import spaces
from gym.utils import seeding
import numpy as np
import json
import cmath
from hardware import hardware_interface
from camera import capture

FPS = 30
# Desired Object Position
TARGET_POS = [23, 10]
GRAVITY = -30

PIX2MM = 300 / 425.0 # Scaling factor
MM2PIX = 1 / PIX2MM

##########################
# Exterior Box Dimension #
##########################
# All dimensions in mm
BOX_WIDTH = 300
BOX_HEIGHT = 250
BOX_HEIGHT_BELOW_ACTUATORS = -1

###################
# Body Dimensions #
###################
# All dimensions in mm
OBJ_SIZE = 40
# OBJ_POS_OFFSET = 30  # should be greater than half the object size
ACTUATOR_TIP_SIZE = 8
# Distance between the wall and the center of the first actuator
BOX_SIDE_OFFSET = 5
LINK_WIDTH = 8
# LINK_HEIGHT = 60
# Do not modify
GAP = (1 - BOX_SIDE_OFFSET * 2) / 4

# ####################
# # Motor Parameters #
# ####################
# MOTOR_SPEED = 10  # m/s
# MOTOR_TORQUE = 80

########################
# Rendering Parameters #
########################
VIEWPORT_W = 640
VIEWPORT_H = 480

ACTUATOR_TRANSLATION_MAX = 0 # hardware_param
ACTUATOR_TRANSLATION_MEAN = ACTUATOR_TRANSLATION_MAX / 2
ACTUATOR_TRANSLATION_AMP = ACTUATOR_TRANSLATION_MAX / 2

#####################
# Camera Parameters #
#####################
CAMERA_CONFIG = capture.CameraConfig()
# Actuator mask parameters
ACTUATOR_X1 = [0, 0, 0, 0, 0]
ACTUATOR_X2 = [CAMERA_CONFIG.frame_width, CAMERA_CONFIG.frame_width,
               CAMERA_CONFIG.frame_width, CAMERA_CONFIG.frame_width, CAMERA_CONFIG.frame_width]
ACTUATOR_Y1 = [0, 0, 0, 0, 0]
ACTUATOR_Y2 = [CAMERA_CONFIG.frame_height, CAMERA_CONFIG.frame_height,
               CAMERA_CONFIG.frame_height, CAMERA_CONFIG.frame_height, CAMERA_CONFIG.frame_height]

ACTUATOR_DELTA = (int)(CAMERA_CONFIG.frame_height / 5)
ACTUATOR_OFFSET = (int)((CAMERA_CONFIG.frame_width - CAMERA_CONFIG.frame_height) / 2)

for i in range(5):
    ACTUATOR_X1[i] = (int)(ACTUATOR_OFFSET + i * ACTUATOR_DELTA)
    ACTUATOR_X2[i] = (int)(ACTUATOR_OFFSET + (i + 1) * ACTUATOR_DELTA)
    ACTUATOR_Y1[i] = 0
    ACTUATOR_Y2[i] = CAMERA_CONFIG.frame_height

CAMERA_CONFIG_FILENAME = 'config/camera_arena.json'
OUC_PARAMS_FILENAME = 'tracking/white_ping.json'
ACTUATOR_PARAMS_FILENAME = 'tracking/blue_actuator.json'

# Dont add more noise to hardware for now, can add later to ensure better stability
# ####################
# # Noise Parameters #
# ####################
# OBJ_POS_STDDEV = BOX_WIDTH / 100.0
# OBJ_VEL_STDDEV = 0  # Nothing set currently
# ACTUATOR_POS_STDDEV = BOX_WIDTH / 100.0
# ACTUATOR_VEL_STDDEV = 0  # Nothing set currently

#################################
# Reward Calculation Parameters #
#################################
MAX_DIST_TO_TARGET = np.sqrt(np.square(BOX_WIDTH) + np.square(BOX_HEIGHT))
# Maximum distance adjacent actuators can be apart vertically due to the membrane
MAX_VERT_DIST_BETWEEN_ACTUATORS = BOX_WIDTH / 4
# Maximum steps at the target before the episode is deemed to be successfully completed
MAX_TARGET_COUNT = 100


class Membrane(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None  # to be used later for rendering

        # Load camera config
        try:
            f_cam_conf = open(CAMERA_CONFIG_FILENAME, 'r')
            CAMERA_CONFIG.from_dict(json.load(f_cam_conf))
            f_cam_conf.close()
            print('Loaded camera config from \'' + CAMERA_CONFIG_FILENAME + '\'')
        except IOError:
            print('Camera config not found, using default config\n')
            # print(json.dumps(camera_config.to_dict(), sort_keys=True, indent=4))

        # Load tracking parameters
        self.ouc_params = capture.TrackParams()
        self.actuator_params = capture.TrackParams()
        try:
            ouc_params_f = open(OUC_PARAMS_FILENAME, 'r')
            self.ouc_params.from_dict(json.load(ouc_params_f))
            ouc_params_f.close()
            print('Loaded OUC params from \'' + OUC_PARAMS_FILENAME + '\'')
        except IOError:
            print('OUC params not found, stopping...\n')
            exit(1)
        try:
            actuator_params_f = open(ACTUATOR_PARAMS_FILENAME, 'r')
            self.actuator_params.from_dict(json.load(actuator_params_f))
            actuator_params_f.close()
            print('Loaded OUC params from \'' + ACTUATOR_PARAMS_FILENAME + '\'')
        except IOError:
            print('OUC params not found, stopping...\n')
            exit(1)

        self.arena_camera = capture.init_camera(CAMERA_CONFIG)

        self.previous_state = [0, 0,
                               0, 0,
                               0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0]

        # World is real
        # self.world = b2World(gravity=[0, GRAVITY], doSleep=True)
        self.exterior_box = None
        # Five linear actuators
        self.actuator_list = []
        # Object to be manipulated
        self.object = None
        # # Linkages
        # if self.with_linkage:
        #     self.link_left_list = []  # four links
        #     self.link_right_list = []  # four links

        # Drawlist for rendering
        self.drawlist = []

        # Observation Space
        # [object posx, object posy, actuator1 pos.y, ... , actuator5 pos.y, actuator1 speed.y, ... , actuator5 speed.y]
        high = np.array([np.inf] * 14)
        self.observation_space = spaces.Box(low=-high, high=high)

        # Continuous action space; one for each linear actuator (5 total)
        # action space represents velocity
        self.action_space = spaces.Box(-1, 1, (5,))
        self.prev_shaping = None  # for reward calculation

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _check_actuator_pos(self):
        result = True
        for i in range(4):
            dist_diff = np.abs(self.actuator_list[i + 1].position.y - self.actuator_list[i].position.y)
            if dist_diff > MAX_VERT_DIST_BETWEEN_ACTUATORS:
                result = False
                break
        return result

    def _destroy(self):
        if not self.exterior_box: return  # return if the exterior box hasn't been created
        # self.world.DestroyBody(self.exterior_box)
        self.exterior_box = None
        # self.world.DestroyBody(self.object)
        self.object = None

        # for actuator in self.actuator_list:
        #     self.world.DestroyBody(actuator)
        self.actuator_list = []
        self.arena_camera.release()

    def _reset(self):
        self._destroy()

        # Perform a hardware reset:
        # Create a v-shape to let the ball roll down
        # Wiggle each actuator to make sure it is at the bottom
        # Move all actuators to a rest position

        # # Creating the Exterior Box that defines the 2D Plane
        # self.exterior_box = self.world.CreateStaticBody(
        #     position=(0, 0),
        #     shapes=b2LoopShape(vertices=EXT_BOX_POLY)
        # )
        # self.exterior_box.color1 = (0, 0, 0)
        # self.exterior_box.color2 = (0, 0, 0)
        #
        # # Creating the object to manipulate
        # object_fixture = b2FixtureDef(
        #     shape=b2CircleShape(radius=BOX_WIDTH * OBJ_SIZE / 2),
        #     density=1,
        #     friction=0.6,
        #     restitution=0.0
        # )

        # Randomizing object's initial position
        # object_position = (
        #     self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_WIDTH-BOX_WIDTH*OBJ_POS_OFFSET),
        #     BOX_HEIGHT/5
        #     )
        # object_position = (
        #     self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_WIDTH-BOX_WIDTH*OBJ_POS_OFFSET),
        #     self.np_random.uniform(BOX_WIDTH*OBJ_POS_OFFSET,BOX_HEIGHT-BOX_WIDTH*OBJ_POS_OFFSET)
        #     )
        object_position = (BOX_WIDTH / 4, BOX_HEIGHT / 3)
        self.object = self.world.CreateDynamicBody(
            position=object_position,
            fixtures=object_fixture,
            linearDamping=0.3  # Control this parameter for surface friction
        )
        self.object.at_target = False
        self.object.at_target_count = 0
        self.object.color1 = (1, 1, 0)
        self.object.color2 = (0, 0, 0)

        # # Creating 5 actuators
        # actuator_fixture = b2FixtureDef(
        #     shape=b2CircleShape(radius=BOX_WIDTH * ACTUATOR_TIP_SIZE / 2),
        #     density=1,
        #     friction=0.6,
        #     restitution=0.0,
        #     groupIndex=-1
        # )

        # for i in range(5):
        #     actuator = self.world.CreateDynamicBody(
        #         position=((BOX_SIDE_OFFSET + GAP * i) * BOX_WIDTH, 0),
        #         fixtures=actuator_fixture
        #     )
        #     actuator.color1 = (0, 0, 0.5)
        #     actuator.color2 = (0, 0, 0)
        #
        #     actuator.joint = self.world.CreatePrismaticJoint(
        #         bodyA=self.exterior_box,
        #         bodyB=actuator,
        #         anchor=actuator.position,
        #         axis=(0, 1),
        #         lowerTranslation=0,
        #         upperTranslation=ACTUATOR_TRANSLATION_MAX,
        #         enableLimit=True,
        #         maxMotorForce=100000.0,
        #         motorSpeed=0,
        #         enableMotor=True
        #     )

            # self.actuator_list.append(actuator)

        self.drawlist = self.actuator_list + [self.object]

        # if self.with_linkage:
        #     # Creating the linkages that will form the semi-flexible membrane
        #     link_fixture = b2FixtureDef(
        #         shape=b2PolygonShape(box=(LINK_WIDTH * BOX_WIDTH / 2, LINK_HEIGHT * BOX_WIDTH / 2)),
        #         density=1,
        #         friction=0.6,
        #         restitution=0.0,
        #         groupIndex=-1  # neg index to prevent collision
        #     )
        #
        #     for i in range(4):
        #         link_left = self.world.CreateDynamicBody(
        #             position=(BOX_WIDTH * (BOX_SIDE_OFFSET + GAP * i + LINK_WIDTH / 2), 0),
        #             fixtures=link_fixture
        #         )
        #         link_left.color1 = (0, 1, 1)
        #         link_left.color2 = (1, 0, 1)
        #         self.link_left_list.append(link_left)
        #
        #         link_right = self.world.CreateDynamicBody(
        #             position=(BOX_WIDTH * (BOX_SIDE_OFFSET + GAP * (i + 1) - LINK_WIDTH / 2), 0),
        #             fixtures=link_fixture
        #         )
        #         link_right.color1 = (0, 1, 1)
        #         link_right.color2 = (1, 0, 1)
        #         self.link_right_list.append(link_right)
        #
        #         joint_left = self.world.CreateRevoluteJoint(
        #             bodyA=self.actuator_list[i],
        #             bodyB=link_left,
        #             anchor=self.actuator_list[i].worldCenter,
        #             collideConnected=False
        #         )
        #
        #         joint_right = self.world.CreateRevoluteJoint(
        #             bodyA=self.actuator_list[i + 1],
        #             bodyB=link_right,
        #             anchor=self.actuator_list[i + 1].worldCenter,
        #             collideConnected=False
        #         )
        #
        #         joint_middle = self.world.CreatePrismaticJoint(
        #             bodyA=link_left,
        #             bodyB=link_right,
        #             anchor=(
        #             link_right.position.x - BOX_WIDTH * (LINK_WIDTH / 2 + LINK_HEIGHT / 2), link_right.position.y),
        #             axis=(1, 0),
        #             lowerTranslation=0,
        #             upperTranslation=BOX_WIDTH * LINK_WIDTH * 2 / 3,
        #             enableLimit=True
        #         )
        #     # Adding linkages to the drawlist
        #     self.drawlist = self.link_left_list + self.link_right_list + self.drawlist

        return self._step(np.array([0, 0, 0, 0, 0]))[0]  # action: zero motor speed

    def _step(self, action):
        # Totally recall previous state
        ouc_x = self.previous_state[0]
        ouc_y = self.previous_state[1]
        actuators_y = self.previous_state[4:9]
        # Set motor speeds from the action outputs
        send_values = [0, 0, 0, 0, 0]
        for i in range(5):
            send_values[i] = float(np.clip(action[i], -1, 1))
        hardware_interface.set_servo_speeds(send_values)

        # Capture an image and process it
        camera_capture = capture.undistort(capture.capture_frame(self.arena_camera),
                                   np.array(CAMERA_CONFIG.camera_matrix), np.array(CAMERA_CONFIG.dist_coefs))
        ouc_list = capture.track_objects(camera_capture, self.ouc_params)
        actuator_list = capture.track_objects(camera_capture, self.actuator_params)

        if len(ouc_list) != 0:
            ouc_main = capture.find_largest(ouc_list)
            # Convert pixels to mm relative to bottom
            ouc_x = ouc_main[0]
            ouc_y = ouc_main[1]
        else:
            pass
            # print('No \'ouc\' found!')

        if len(actuator_list) != 0:
            for i in range(5):
                index = 4 - i
                temp = capture.find_largest_in_area(actuator_list, ACTUATOR_X1[index], ACTUATOR_Y1[index],
                                            ACTUATOR_X2[index], ACTUATOR_Y2[index])
                if temp != ():
                    actuators_y[index] = temp[1]
                    actuator_list.remove(temp)
        else:
            pass
            # print('No actuators found')

        # All measurements are now in mm
        # Calculate velocity of actuators using the last n-values

        # Set the state


        # # Move forward one frame
        # self.world.Step(1.0 / FPS, 10, 10)

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
            (object_pos[0] - BOX_WIDTH / 2) / (BOX_WIDTH / 2),
            (object_pos[1] - BOX_HEIGHT / 2) / (BOX_HEIGHT / 2),
            object_vel[0] / ((BOX_WIDTH / 2) * FPS),
            object_vel[1] / ((BOX_HEIGHT / 2) * FPS),
            (actuator_pos[0] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[1] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[2] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[3] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP,
            (actuator_pos[4] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP,
            (actuator_vel[0]) / MOTOR_SPEED,
            (actuator_vel[1]) / MOTOR_SPEED,
            (actuator_vel[2]) / MOTOR_SPEED,
            (actuator_vel[3]) / MOTOR_SPEED,
            (actuator_vel[4]) / MOTOR_SPEED,
        ]
        assert len(state) == 14

        # Rewards
        # dist_to_target = TARGET_POS[0]-self.object.position.x
        dist_to_target = np.sqrt(np.square(TARGET_POS[0] - object_pos[0]) + np.square(TARGET_POS[1] - object_pos[1]))
        reward = 0
        shaping = \
            -150 * dist_to_target / MAX_DIST_TO_TARGET \
            - 150 * np.sqrt(state[2] * state[2] + state[3] * state[3])

        # Check if the objects at target position
        if dist_to_target < 1:
            self.object.at_target = True
            self.object.at_target_count += 1
            shaping += 20
        else:
            self.object.at_target = False
            self.object.at_target_count = 0

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Reduce reward for using the motor
        for a in action:
            reward -= 0.05 * np.clip(np.abs(a), 0, 1)

        # Reward for staying at target position
        reward += 50 * self.object.at_target_count / MAX_TARGET_COUNT

        done = False
        np.abs(self.actuator_list[1].position.y - self.actuator_list[0].position.y)
        if not self._check_actuator_pos():
            done = True
            reward -= 100

        # If object is at the target position the task is complete
        obj_vel_magnitude = np.abs(np.sqrt(np.square(object_vel[0]) + np.square(object_vel[1])))
        if dist_to_target < 1 and obj_vel_magnitude < 1 and self.object.at_target_count >= MAX_TARGET_COUNT:
            done = True
            reward += 100

        self.previous_state = state

        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        pass

        # Draw OUC position

        # Draw actuator positions

        # Draw actuator bounding boxes

        # Draw target position

        # # Draw UI
        # # OUC
        # draw_crosshair(camera_capture, ouc_x, ouc_y, (0, 255, 0), camera_config)
        # draw_crosshair(camera_capture, ouc_top_main[0], ouc_top_main[1], (128, 0, 0), camera_config, 0.5)
        # draw_crosshair(camera_capture, ouc_bottom_main[0], ouc_bottom_main[1], (0, 0, 128), camera_config, 0.5)
        #
        # # Actuators
        # for i in range(num_actuators):
        #     draw_crosshair(camera_capture, actuators_x[i], actuators_y[i], (0, 255, 0), camera_config, 0.5)
        #     draw_box(camera_capture, actuator_x1[i], actuator_y1[i], actuator_x2[i], actuator_y2[i],
        #              (128, 128, 0), 2)
        #
        # text = 'Angle: {0.real:.1f}'.format(ouc_angle)
        #
        # cv2.putText(camera_capture, text, (0, 30), 2, 1, (0, 255, 0), 2)
        #
        # cv2.imshow('img', camera_capture)



        # if close:
        #     if self.viewer is not None:
        #         self.viewer.close()
        #         self.viewer = None
        #     return
        #
        # from gym.envs.classic_control import rendering
        #
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        #     self.viewer.set_bounds(-5, BOX_WIDTH + 5, -5 - BOX_HEIGHT_BELOW_ACTUATORS, BOX_HEIGHT + 5)
        #
        # # Actuator start position visualized
        # self.viewer.draw_polyline([(0, 0), (BOX_WIDTH, 0)], color=(1, 0, 1))
        #
        # # Target Position Visualized
        # self.viewer.draw_polyline([(TARGET_POS[0], 0), (TARGET_POS[0], BOX_HEIGHT)], color=(1, 0, 0))
        # self.viewer.draw_polyline([(0, TARGET_POS[1]), (BOX_WIDTH, TARGET_POS[1])], color=(1, 0, 0))
        #
        # # Exterior Box Visualized
        # box_fixture = self.exterior_box.fixtures[0]
        # box_trans = box_fixture.body.transform
        # box_path = [box_trans * v for v in box_fixture.shape.vertices]
        # box_path.append(box_path[0])
        # self.viewer.draw_polyline(box_path, color=self.exterior_box.color2, linewidth=2)
        #
        # for obj in self.drawlist:
        #     for f in obj.fixtures:
        #         trans = f.body.transform
        #         if type(f.shape) is b2CircleShape:
        #             t = rendering.Transform(translation=trans * f.shape.pos)
        #             self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
        #             self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
        #         else:
        #             path = [trans * v for v in f.shape.vertices]
        #             self.viewer.draw_polygon(path, color=obj.color1)
        #             path.append(path[0])
        #             self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        #
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class MembraneWithoutLinkages(Membrane):
    with_linkage = False


if __name__ == "__main__":
    env = Membrane()
    s = env.reset()

