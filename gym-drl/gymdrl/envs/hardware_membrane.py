# HARDWARE MEMBRANE ENVIRONMENT
#
# Copyright (c) 2017 William Choi, Alex Kyriazis, Ivan Zinin; all rights reserved

import gym
import cv2
from gym import spaces
from gym.utils import seeding
import numpy as np
import json
import time
from gymdrl.envs.hardware import hardware_interface
from gymdrl.envs.camera import capture

epsilon = 1E-4 # used for float comparisons

PIX2MM = 300 / 425.0  # Scaling factor
MM2PIX = 1 / PIX2MM

# Desired Object Position from bottom left corner
TARGET_POS = [210, 50]  # in mm

VIEWPORT_W = 640
VIEWPORT_H = 480

##########################
# Exterior Box Dimension #
##########################
# All dimensions in mm
BOX_WIDTH = 267
BOX_HEIGHT = 300
BOX_TRIM_BOTTOM = 39
BOX_TRIM_LEFT = 92

# UI PARAMETERS:
BOX_UI_X1 = int(MM2PIX * (BOX_TRIM_LEFT))
BOX_UI_X2 = int(MM2PIX * (BOX_TRIM_LEFT + BOX_WIDTH))
BOX_UI_Y2 = int(VIEWPORT_H - MM2PIX * (BOX_TRIM_BOTTOM))
BOX_UI_Y1 = int(VIEWPORT_H - MM2PIX * (BOX_TRIM_BOTTOM + BOX_HEIGHT))

BOX_UI_CENTER_X = int(MM2PIX * (BOX_TRIM_LEFT + BOX_WIDTH / 2))
BOX_UI_CENTER_Y = int(VIEWPORT_H - MM2PIX * (BOX_TRIM_BOTTOM + BOX_HEIGHT/2))

#######################
# Hardware parameters #
#######################
# All dimensions in mm
# OBJ_SIZE = 40

ACTUATOR_TRANSLATION_MAX = 65  # hardware_param
ACTUATOR_TRANSLATION_MEAN = ACTUATOR_TRANSLATION_MAX / 2
ACTUATOR_TRANSLATION_AMP = ACTUATOR_TRANSLATION_MAX / 2

MAX_SPEED = 160  # mm/s assumed maximum motor speed

########################
# Rendering Parameters #
########################

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

ACTUATOR_DELTA = int(CAMERA_CONFIG.frame_height / 5)
ACTUATOR_OFFSET = int((CAMERA_CONFIG.frame_width - CAMERA_CONFIG.frame_height) / 2)

for i in range(5):
    ACTUATOR_X1[i] = int(ACTUATOR_OFFSET + i * ACTUATOR_DELTA)
    ACTUATOR_X2[i] = int(ACTUATOR_OFFSET + (i + 1) * ACTUATOR_DELTA)
    ACTUATOR_Y1[i] = 0
    ACTUATOR_Y2[i] = CAMERA_CONFIG.frame_height

CONFIG_PREFIX = '../gym-drl/gymdrl/envs/camera/'
CAMERA_CONFIG_FILENAME = CONFIG_PREFIX + 'config/camera_arena.json'
OUC_PARAMS_FILENAME = CONFIG_PREFIX + 'tracking/white_ping.json'
ACTUATOR_PARAMS_FILENAME = CONFIG_PREFIX + 'tracking/blue_actuator.json'

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
MAX_VERT_DIST_BETWEEN_ACTUATORS = ACTUATOR_TRANSLATION_MAX
# Maximum steps at the target before the episode is deemed to be successfully completed
MAX_TARGET_COUNT = 100
# Minimum distance between target and OUC to be considered at the target location
MIN_DIST_TO_TARGET = 10  # mm


class MembraneHardware(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self):
        self.count = 0
        self._seed()
        self.viewer = None  # to be used later for rendering

        self.prev_state = None
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

        self.camera_capture = None

        # Initialize hardware devices
        self.arena_camera = capture.init_camera(CAMERA_CONFIG)
        capture.camera_skip_frames(self.arena_camera, CAMERA_CONFIG)
        self.serial = hardware_interface.init_serial()

        zero_state = [0, 0,
                      0, 0, 0, 0, 0]

        self.previous_pos = zero_state

        # Observation Space
        # [object posx, object posy, actuator1 pos.y, ... , actuator5 pos.y, actuator1 speed.y, ... , actuator5 speed.y]
        high = np.array([np.inf] * 14)
        self.observation_space = spaces.Box(low=-high, high=high)

        # Continuous action space; one for each linear actuator (5 total)
        # action space represents velocity
        self.action_space = spaces.Box(-1, 1, (5,))
        self.prev_shaping = None  # for reward calculation

        self.time_previous = None

        self.ouc_ui_pos = (0, 0)
        self.actuators_ui_pos = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def _check_actuator_pos(self):
    #     result = True
    #     for i in range(4):
    #         dist_diff = np.abs(self.actuator_list[i + 1].position.y - self.actuator_list[i].position.y)
    #         if dist_diff > MAX_VERT_DIST_BETWEEN_ACTUATORS:
    #             result = False
    #             break
    #     return result

    def _destroy(self):
        self.arena_camera.release()
        cv2.destroyAllWindows()

    def _reset(self):

        # Perform a hardware reset:
        # Create a v-shape to let the ball roll down
        # Wiggle each actuator to make sure it is at the bottom
        # Move all actuators to a rest position

        self.object_at_target = False
        self.object_at_target_count = 0

        # Set lowest position for actuators
        # reset_values = [125, 105, 90, 105, 125]

        # Reset used to randomize the new position
        reset_speeds = [0.5 + np.random.rand() / 2,
                        0.5 + np.random.rand() / 2,
                        0.5 + np.random.rand() / 2,
                        0.5 + np.random.rand() / 2,
                        0.5 + np.random.rand() / 2]
        hardware_interface.set_servo_speeds(self.serial, reset_speeds)
        time.sleep(0.5)
        reset_values = [90 + np.random.rand() * 40,
                        90 + np.random.rand() * 40,
                        90 + np.random.rand() * 40,
                        90 + np.random.rand() * 40,
                        90 + np.random.rand() * 40]
        hardware_interface.set_servo_angles(self.serial, reset_values)
        time.sleep(0.5)

        self.time_previous = time.time()

        return self._step(np.array([0, 0, 0, 0, 0]))[0]  # action: zero motor speed

    def _step(self, action):

        # Uncomment to use a programmed policy
        # if self.prev_state is not None:
        #     action = self.programmed_policy(self.prev_state)

        self.count = self.count + 1
        # Totally recall previous state
        ouc_x = self.previous_pos[0]
        ouc_y = self.previous_pos[1]
        actuators_y = self.previous_pos[2:7]

        # Capture an image and process it
        self.camera_capture = capture.undistort(capture.capture_frame(self.arena_camera),
                                                np.array(CAMERA_CONFIG.camera_matrix),
                                                np.array(CAMERA_CONFIG.dist_coefs))
        ouc_list = capture.track_objects(self.camera_capture, self.ouc_params)
        actuator_list = capture.track_objects(self.camera_capture, self.actuator_params)

        if len(ouc_list) != 0:
            ouc_main = capture.find_largest(ouc_list)
            self.ouc_ui_pos = (ouc_main[0], ouc_main[1])
            # Convert pixels to mm relative to bottom left arena corner
            ouc_x = ouc_main[0] * PIX2MM - BOX_TRIM_LEFT
            ouc_y = (VIEWPORT_H - ouc_main[1]) * PIX2MM - BOX_TRIM_BOTTOM
        else:
            pass
            # print('No \'ouc\' found!')

        if len(actuator_list) != 0:
            for i in range(5):
                index = 4 - i
                temp = capture.find_largest_in_area(actuator_list, ACTUATOR_X1[index], ACTUATOR_Y1[index],
                                                    ACTUATOR_X2[index], ACTUATOR_Y2[index])
                if temp != ():
                    self.actuators_ui_pos[index] = (temp[0], temp[1])
                    actuators_y[index] = (VIEWPORT_H - temp[1]) * PIX2MM - BOX_TRIM_BOTTOM
                    actuator_list.remove(temp)
        else:
            pass
            # print('No actuators found')

        # All measurements are now in mm from the lower left corner
        # Calculate velocity of actuators using the previous value
        current_time = time.time()
        delta_t = current_time - self.time_previous
        self.time_previous = current_time
        # print('Freq: ' + (1/delta_t).__str__())

        # Velocities are in mm/s
        object_vel = [
            (ouc_x - self.previous_pos[0]) / delta_t,
            (ouc_y - self.previous_pos[1]) / delta_t,
        ]
        actuator_vel = [
            (actuators_y[0] - self.previous_pos[2]) / delta_t,
            (actuators_y[1] - self.previous_pos[3]) / delta_t,
            (actuators_y[2] - self.previous_pos[4]) / delta_t,
            (actuators_y[3] - self.previous_pos[5]) / delta_t,
            (actuators_y[4] - self.previous_pos[6]) / delta_t
        ]

        self.previous_pos = [ouc_x,
                             ouc_y,
                             actuators_y[0],
                             actuators_y[1],
                             actuators_y[2],
                             actuators_y[3],
                             actuators_y[4]]

        # Observation space (state)
        state = [
            np.clip((ouc_x - BOX_WIDTH / 2) / (BOX_WIDTH / 2), -1, 1),
            np.clip((ouc_y - BOX_HEIGHT / 2) / (BOX_HEIGHT / 2), -1, 1),
            np.clip(object_vel[0] / MAX_SPEED, -1, 1),
            np.clip(object_vel[1] / MAX_SPEED, -1, 1),
            np.clip((actuators_y[0] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP, -1, 1),
            np.clip((actuators_y[1] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP, -1, 1),
            np.clip((actuators_y[2] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP, -1, 1),
            np.clip((actuators_y[3] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP, -1, 1),
            np.clip((actuators_y[4] - ACTUATOR_TRANSLATION_MEAN) / ACTUATOR_TRANSLATION_AMP, -1, 1),
            np.clip(actuator_vel[0] / MAX_SPEED, -1, 1),
            np.clip(actuator_vel[1] / MAX_SPEED, -1, 1),
            np.clip(actuator_vel[2] / MAX_SPEED, -1, 1),
            np.clip(actuator_vel[3] / MAX_SPEED, -1, 1),
            np.clip(actuator_vel[4] / MAX_SPEED, -1, 1)
        ]
        self.prev_state = state

        assert len(state) == 14

        # For debug puroposes:
        # print('OUC pos: {:.2f},{:.2f}'.format(state[0], state[1]))
        # print('actuator vel: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(state[9], state[10], state[11], state[12], state[13]))
        # print('actuator pos: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(state[4], state[5], state[6], state[7], state[8]))

        # Set motor speeds from the action outputs
        send_values = [0, 0, 0, 0, 0]
        for i in range(5):
            send_values[i] = float(np.clip(action[i], -1.0, 1.0))
        # Now make sure that if we are at the edge of the actuator movement, we don't actually move it more:
        for i in range(5):
            if abs(state[4+i] - 1) < epsilon:
                send_values[i] = np.clip([send_values[i]], -1.0, 0.0)[0]
                continue
            if abs(state[4+i] + 1) < epsilon:
                send_values[i] = np.clip([send_values[i]], 0.0, 1.0)[0]

        # Temporary speed reduction until retrain
        send_values = np.array(send_values) / 3.0
        # print('Send speeds: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(
        #     send_values[0], send_values[1], send_values[2], send_values[3], send_values[4]))
        hardware_interface.set_servo_speeds(self.serial, send_values)

        # Rewards
        # dist_to_target = TARGET_POS[0]-self.object.position.x
        # dist_to_target = np.sqrt(np.square(TARGET_POS[0] - ouc_x) + np.square(TARGET_POS[1] - ouc_y))

        reward = 0
        shaping = \
            - 200 * np.abs(TARGET_POS[0] - ouc_x) / BOX_WIDTH \
            - 200 * np.abs(TARGET_POS[1] - ouc_y) / BOX_HEIGHT \
            - 50 * np.abs(state[2]) \
            - 50 * np.abs(state[3])

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        if (np.abs(ouc_x - TARGET_POS[0])) < 15:
            if (np.abs(ouc_y - TARGET_POS[1])) < 15:
                reward += 50
        # Reduce reward for using the motor
        for a in action:
            reward -= 1 * np.clip(np.abs(a), 0, 1)

        done = False

        return np.array(state), reward, done, {}

    def _render(self, mode='human', close=False):
        if self.camera_capture is not None:
            # Draw OUC position
            capture.draw_crosshair(self.camera_capture, self.ouc_ui_pos[0], self.ouc_ui_pos[1], (0, 255, 0), CAMERA_CONFIG)

            # Draw actuator positions
            for i in range(5):
                capture.draw_crosshair(self.camera_capture, self.actuators_ui_pos[i][0], self.actuators_ui_pos[i][1],
                                       (0, 255, 0), CAMERA_CONFIG, 0.5)
                # Draw actuator bounding boxes
                capture.draw_box(self.camera_capture, ACTUATOR_X1[i], ACTUATOR_Y1[i], ACTUATOR_X2[i], ACTUATOR_Y2[i],
                                 (200, 0, 0), 1)

            # Draw arena
            capture.draw_box(self.camera_capture, BOX_UI_X1, BOX_UI_Y1, BOX_UI_X2, BOX_UI_Y2, (20, 20, 20), 1)
            capture.draw_crosshair(self.camera_capture, BOX_UI_CENTER_X, BOX_UI_CENTER_Y, (20, 20, 20), CAMERA_CONFIG,
                                   0.5, make_label=False)

            # Draw target position
            target_x = int((TARGET_POS[0] + BOX_TRIM_LEFT)* MM2PIX)
            target_y = VIEWPORT_H - int((TARGET_POS[1] + BOX_TRIM_BOTTOM) * MM2PIX)
            capture.draw_crosshair(self.camera_capture, target_x, target_y, (0, 0, 255), CAMERA_CONFIG, make_label=False)

            cv2.imshow('Hardware Membrane', self.camera_capture)
            cv2.waitKey(10)

    def programmed_policy(self, state):

        FAST_SPEED = 1
        MEDIUM_SPEED = 0.5
        SLOW_SPEED = 0.1


        ACTUATOR_START= (BOX_WIDTH - 60*4 ) / 2
        ACTUATOR_SPACING = 60

        act_pos = [state[i] for i in range(6,11)]

        p = (self.previous_pos[0]-ACTUATOR_START)/ACTUATOR_SPACING
        action = -MEDIUM_SPEED*np.ones(5)

        if (TARGET_POS[0]-self.previous_pos[0]) > 0:
            #move right
            b=int(np.floor(p))
            action[b] = SLOW_SPEED*2
            if b-1 >=0:
                action[b-1] = SLOW_SPEED * 3
        else:
            #move left
            action[int(np.ceil(p))] = SLOW_SPEED*2
            if int(np.ceil(p)) == 4:
                action[int(np.ceil(p))] = SLOW_SPEED

        if self.previous_pos[0] > BOX_WIDTH/2:
            action[2]=MEDIUM_SPEED


        return action

if __name__ == "__main__":
    env = MembraneHardware()
    s = env.reset()
