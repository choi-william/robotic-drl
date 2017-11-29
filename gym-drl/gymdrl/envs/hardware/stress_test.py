# from hardware.hardware_interface import set_servo_angles
from hardware_interface import set_servo_angles
from cmath import sin
import time

MIN_ANGLE_INPUT = 0
MAX_ANGLE_INPUT = 180
send_angles = [100, 100, 100, 100, 100]
DEG2RAD = 3.14159 / 180
FREQ = 5

#
# Arena parameters
#
# For now assume all actuators are the same
MAX_ACTUATOR_ANGLE = 160
MIN_ACTUATOR_ANGLE = 90
#
# Normal operation parameters
#

# Sine Wave
SINE_ITERATIONS = 6000*FREQ
SINE_OFFSET = (MAX_ACTUATOR_ANGLE + MIN_ACTUATOR_ANGLE)/2  # degrees
SINE_AMPLITUDE = 15  # in degrees
SINE_DELAY = -1.5  # in seconds
SINE_SPEED = 3

#
# Vibration test parameters
#
VIBRATION_ITERATIONS = 60*FREQ
VIBRATION_OFFSET = (MAX_ACTUATOR_ANGLE + MIN_ACTUATOR_ANGLE)/2  # degrees
VIBRATION_AMPLITUDES = [5, 4]  # in degrees
VIBRATION_DELAY = [0.1, 0.05, 0.05, 0.05, 0.05]  # in seconds

#
# Slamming test parameters
#
SLAMMING_ITERATIONS = 15*FREQ
SLAMMING_DELAY = 1


def xor(a, b):
    return (a and not b) or (b and not a)


def sine_test():
    for iteration in range(SINE_ITERATIONS):
        for actuator in range(5):
            send_angles[actuator] = SINE_OFFSET + \
                int(SINE_AMPLITUDE * sin((DEG2RAD * iteration + actuator * SINE_DELAY) * SINE_SPEED).real)
        set_servo_angles(send_angles)
        time.sleep(0.01)


# Vibrates actuators in a 'w' shape to minimise arena movement
def vibration_test():
    for test_num in range(len(VIBRATION_AMPLITUDES)):
        amplitude = VIBRATION_AMPLITUDES[test_num]
        delay = VIBRATION_DELAY[test_num]
        print('Amplitude: ' + amplitude.__str__())
        for iteration in range(VIBRATION_ITERATIONS):
            for actuator in range(5):
                # python doesen't have xor :(
                if xor(actuator % 2 is 0, iteration % 2 is 0):
                    send_angles[actuator] = VIBRATION_OFFSET
                else:
                    send_angles[actuator] = VIBRATION_OFFSET + amplitude

            set_servo_angles(send_angles)
            time.sleep(delay)


def slamming_test():
    for iteration in range(SLAMMING_ITERATIONS):
        for actuator in range(5):
            if iteration % 2 is 0:
                send_angles[actuator] = MIN_ACTUATOR_ANGLE
            else:
                send_angles[actuator] = MAX_ACTUATOR_ANGLE

        set_servo_angles(send_angles)
        time.sleep(SLAMMING_DELAY)


if __name__ == '__main__':
    print('Starting stress test')
    #
    # print('Starting sine test')
    # sine_test()
    # print('Finished sine test')
    #
    # time.sleep(1)
    #
    print('Starting vibration test')
    vibration_test()
    print('Finished vibration test')

    # time.sleep(1)
    #
    # print('Starting slamming test')
    # slamming_test()
    # print('Finished slamming test')

    print('Stress test complete')
