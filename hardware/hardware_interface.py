import serial

MIN_SPEED_INPUT = -1;
MAX_SPEED_INPUT = 1;

MIN_ANGLE_INPUT = 0;
MAX_ANGLE_INPUT = 180;

PORT = '/dev/tty.usbserial'
BAUD_RATE = 115200
ser = serial.Serial(PORT, BAUD_RATE)

def set_servo_speeds(servo_speeds):
	S_CHAR = ord('S')
	encoded_speed_bytes =  enconde_bytes(servo_speeds,MIN_SPEED_INPUT,MAX_SPEED_INPUT)
	ser.write(bytearray([S_CHAR]+encoded_speed_bytes))

def set_servo_angles(servo_angles):
	A_CHAR = ord('A')
	encoded_angle_bytes =  enconde_bytes(servo_angles,MIN_ANGLE_INPUT,MAX_ANGLE_INPUT)
	ser.write(bytearray([A_CHAR]+encoded_angle_bytes))

def enconde_bytes(values,min_value,max_value)
	return round(255*(values-min_value)/(max_value-min_value))



