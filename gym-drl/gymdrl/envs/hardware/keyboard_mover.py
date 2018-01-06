import getch
import numpy as np
from gymdrl.envs.hardware.hardware_interface import set_servo_angles
from gymdrl.envs.hardware.hardware_interface import set_servo_speeds
from gymdrl.envs.hardware.hardware_interface import init_serial

# Keyboard controls with corresponding key codes (not actually needed lol):
#  Q   W   E   R   T
# 113 119 101 114 116
#  A   S   D   F   G
#  97 115 100 102 103

control_speed = True

SET_ANGLE_FACTOR = 1
SET_SPEED_FACTOR = 0.1
PLUS_SPEED = 10
STOP_VAR = 3.1415

actuator_map = {'q': (0, True, 1),
                'w': (1, True, 1),
                'e': (2, True, 1),
                'r': (3, True, 1),
                't': (4, True, 1),
                'a': (0, True, -1),
                's': (1, True, -1),
                'd': (2, True, -1),
                'f': (3, True, -1),
                'g': (4, True, -1),

                'Q': (0, True, PLUS_SPEED),
                'W': (1, True, PLUS_SPEED),
                'E': (2, True, PLUS_SPEED),
                'R': (3, True, PLUS_SPEED),
                'T': (4, True, PLUS_SPEED),
                'A': (0, True, -PLUS_SPEED),
                'S': (1, True, -PLUS_SPEED),
                'D': (2, True, -PLUS_SPEED),
                'F': (3, True, -PLUS_SPEED),
                'G': (4, True, -PLUS_SPEED),

                'z': (0, True, STOP_VAR),
                'x': (1, True, STOP_VAR),
                'c': (2, True, STOP_VAR),
                'v': (3, True, STOP_VAR),
                'b': (4, True, STOP_VAR),
                'Z': (0, True, STOP_VAR),
                'X': (1, True, STOP_VAR),
                'C': (2, True, STOP_VAR),
                'V': (3, True, STOP_VAR),
                'B': (4, True, STOP_VAR),

                'u': (0, False, 1),
                'i': (1, False, 1),
                'o': (2, False, 1),
                'p': (3, False, 1),
                '[': (4, False, 1),
                'j': (0, False, -1),
                'k': (1, False, -1),
                'l': (2, False, -1),
                ';': (3, False, -1),
                '\'': (4, False, -1),

                'U': (0, False, PLUS_SPEED),
                'I': (1, False, PLUS_SPEED),
                'O': (2, False, PLUS_SPEED),
                'P': (3, False, PLUS_SPEED),
                '{': (4, False, PLUS_SPEED),
                'J': (0, False, -PLUS_SPEED),
                'K': (1, False, -PLUS_SPEED),
                'L': (2, False, -PLUS_SPEED),
                ':': (3, False, -PLUS_SPEED),
                '\"': (4, False, -PLUS_SPEED),

                'n': (0, False, STOP_VAR),
                'm': (1, False, STOP_VAR),
                ',': (2, False, STOP_VAR),
                '.': (3, False, STOP_VAR),
                '/': (4, False, STOP_VAR),
                'N': (0, False, STOP_VAR),
                'M': (1, False, STOP_VAR),
                '<': (2, False, STOP_VAR),
                '>': (3, False, STOP_VAR),
                '?': (4, False, STOP_VAR),
                }

reset_angles = [180, 180, 180, 180, 180]
reset_speeds = [0.0, 0.0, 0.0, 0.0, 0.0]
send_angles = [100, 100, 100, 100, 100]
send_speeds = [0.0, 0.0, 0.0, 0.0, 0.0]
smol_speeds = [0.1, 0.1, 0.1, 0.1, 0.1]
MIN_ANGLE_INPUT = 0
MAX_ANGLE_INPUT = 180

MIN_SPEED_INPUT = -1.0
MAX_SPEED_INPUT = 1.0

spd = 5

if __name__ == '__main__':

    ser = init_serial()
    if control_speed:
        while True:
            key = getch.getch()
            if key == '\n':
                continue
            if key == '0':
                send_angles[0] += 60
                send_angles[1] += 60
                set_servo_angles(ser, send_angles)
                continue
            if key == '=':
                set_servo_angles(ser, reset_angles)
                continue
            if key == '1':
                set_servo_speeds(ser, smol_speeds)
                print(smol_speeds)
                continue
            if key == '!':
                set_servo_speeds(ser, -np.array(smol_speeds))
                print(-np.array(smol_speeds))
                continue
            if key == '2':
                set_servo_speeds(ser, np.array(smol_speeds)*spd)
                print(np.array(smol_speeds)*spd)
                continue
            if key == '@':
                set_servo_speeds(ser, np.array(smol_speeds)*-spd)
                print(np.array(smol_speeds)*-spd)
                continue
            if key == '`':
                set_servo_speeds(ser, reset_speeds)
                print(reset_speeds)
                continue
            try:
                actuator, do_set_pos, direction = actuator_map[key]
            except KeyError:
                continue
            if do_set_pos:
                if direction == STOP_VAR:
                    send_speeds[actuator] = 0.0
                else:
                    vel = direction * SET_SPEED_FACTOR
                    send_speeds[actuator] += vel
                    if send_speeds[actuator] > MAX_SPEED_INPUT:
                        send_speeds[actuator] = MAX_SPEED_INPUT
                    elif send_speeds[actuator] < MIN_SPEED_INPUT:
                        send_speeds[actuator] = MIN_SPEED_INPUT
                print(send_speeds)
                set_servo_speeds(ser, send_speeds)
            else:
                if direction == STOP_VAR:
                    send_angles[actuator] = 100
                else:
                    pos = direction * SET_ANGLE_FACTOR
                    send_angles[actuator] += pos
                    if send_angles[actuator] > MAX_ANGLE_INPUT:
                        send_angles[actuator] = MAX_ANGLE_INPUT
                    elif send_angles[actuator] < MIN_ANGLE_INPUT:
                        send_angles[actuator] = MIN_ANGLE_INPUT
                set_servo_angles(ser, send_angles)
                print(send_angles)
