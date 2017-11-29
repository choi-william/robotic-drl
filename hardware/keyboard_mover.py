import getch
# from hardware.hardware_interface import set_servo_angles
from hardware_interface import set_servo_angles

# Keyboard controls with corresponding key codes (not actually needed lol):
#  Q   W   E   R   T
# 113 119 101 114 116
#  A   S   D   F   G
#  97 115 100 102 103

speed = 1
plus_speed = 10
actuator_map = {'q':(0, 1),
                'w':(1, 1),
                'e':(2, 1),
                'r':(3, 1),
                't':(4, 1),
                'a':(0, -1),
                's':(1, -1),
                'd':(2, -1),
                'f':(3, -1),
                'g':(4, -1),

                'Q': (0, plus_speed),
                'W': (1, plus_speed),
                'E': (2, plus_speed),
                'R': (3, plus_speed),
                'T': (4, plus_speed),
                'A': (0, -plus_speed),
                'S': (1, -plus_speed),
                'D': (2, -plus_speed),
                'F': (3, -plus_speed),
                'G': (4, -plus_speed)}

send_angles = [100, 100, 100, 100, 100]
MIN_ANGLE_INPUT = 0
MAX_ANGLE_INPUT = 180


if __name__ == '__main__':
    while (True):
        key = getch.getch()
        if key == '\n':
            continue
        if key == 'p':
            send_angles[0] += 60
            send_angles[1] += 60
            set_servo_angles(send_angles)
            continue
        try:
            actuator, dir = actuator_map[key]
        except KeyError:
            continue
        pos = dir * speed
        send_angles[actuator] += pos
        if send_angles[actuator] > MAX_ANGLE_INPUT:
            send_angles[actuator] = MAX_ANGLE_INPUT
        elif send_angles[actuator] < MIN_ANGLE_INPUT:
            send_angles[actuator] = MIN_ANGLE_INPUT
        set_servo_angles(send_angles)
