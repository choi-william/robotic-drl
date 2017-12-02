import serial

# Initialize serial port
PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

# Constants
S_CHAR = ord('S')
A_CHAR = ord('A')
NEWLINE_CHAR = ord('\n')

MIN_SPEED_INPUT = -1
MAX_SPEED_INPUT = 1

MIN_ANGLE_INPUT = 0
MAX_ANGLE_INPUT = 180


def init_serial():
    return serial.Serial(PORT, BAUD_RATE)


# sample input: [0.5, 0, 1, -0.2, 0.3] MUST BE BETWEEN MIN_SPEED_INPUT AND MAX_SPEED_INPUT
def set_servo_speeds(ser, servo_speeds):
    # print(servo_speeds)
    encoded_speed_bytes = encode_bytes(servo_speeds, MIN_SPEED_INPUT, MAX_SPEED_INPUT)
    command = [S_CHAR] + encoded_speed_bytes + [NEWLINE_CHAR]
    # print(command)
    ser.write(bytearray(command))


def set_servo_angles(ser, servo_angles):
    # print(servo_angles)
    encoded_angle_bytes = encode_bytes(servo_angles, MIN_ANGLE_INPUT, MAX_ANGLE_INPUT)
    command = [A_CHAR] + encoded_angle_bytes + [NEWLINE_CHAR]
    # print(command)
    ser.write(bytearray(command))


def encode_bytes(values, min_value, max_value):
    return [int(254 * (x - min_value) / (max_value - min_value)) for x in values]
