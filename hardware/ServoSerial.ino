#include < Servo.h >

#define ACTUATOR_1_PIN 3
#define ACTUATOR_2_PIN 5
#define ACTUATOR_3_PIN 6
#define ACTUATOR_4_PIN 9
#define ACTUATOR_5_PIN 10

#define HIGH_ANGLE 180
#define LOW_ANGLE 0

#define BIT_TO_ANGLE_FACTOR 180.0/255.0
#define MID_BIT 127.5

Servo servo_1;
Servo servo_2;
Servo servo_3;
Servo servo_4;
Servo servo_5;

byte inputByte;
int bufferSize = 6;
byte inputBuffer[bufferSize];
bool bufferReady = false;
int bufferIndex = 0;

void setup() {
  servo_1.attach(ACTUATOR_1_PIN);
  servo_2.attach(ACTUATOR_2_PIN);
  servo_3.attach(ACTUATOR_3_PIN);
  servo_4.attach(ACTUATOR_4_PIN);
  servo_5.attach(ACTUATOR_5_PIN);

  // initialize serial:
  Serial.begin(115200);
  while (!Serial);
}

void loop() {
  // Use the 'S' character to represent a servo movement with the following byte as the value in degrees
  if (bufferReady) {
    bufferIndex = 0;
    bufferReady = false;
    if (inputBuffer[0] == 'S') {

      //between -255 and 255
      decoded_speed1 = round(2*(inputBuffer[1]-MID_BIT)); 
      decoded_speed2 = round(2*(inputBuffer[2]-MID_BIT));
      decoded_speed3 = round(2*(inputBuffer[3]-MID_BIT));
      decoded_speed4 = round(2*(inputBuffer[4]-MID_BIT));
      decoded_speed5 = round(2*(inputBuffer[5]-MID_BIT));

      //either 180 or 0
      target_angle1 = (decoded_speed1 > 0) ? HIGH_ANGLE : LOW_ANGLE
      target_angle2 = (decoded_speed2 > 0) ? HIGH_ANGLE : LOW_ANGLE
      target_angle3 = (decoded_speed3 > 0) ? HIGH_ANGLE : LOW_ANGLE
      target_angle4 = (decoded_speed4 > 0) ? HIGH_ANGLE : LOW_ANGLE
      target_angle5 = (decoded_speed5 > 0) ? HIGH_ANGLE : LOW_ANGLE

      servo_1.slowmove(target_angle1,abs(decoded_speed1));
      servo_2.slowmove(target_angle2,abs(decoded_speed2));
      servo_3.slowmove(target_angle3,abs(decoded_speed3));
      servo_4.slowmove(target_angle4,abs(decoded_speed4));
      servo_5.slowmove(target_angle5,abs(decoded_speed5));
    }
    if (inputBuffer[0] == 'A') {
      servo_1.write(inputBuffer[1]*BIT_TO_ANGLE_FACTOR);
      servo_2.write(inputBuffer[2]*BIT_TO_ANGLE_FACTOR);
      servo_3.write(inputBuffer[3]*BIT_TO_ANGLE_FACTOR);
      servo_4.write(inputBuffer[4]*BIT_TO_ANGLE_FACTOR);
      servo_5.write(inputBuffer[5]*BIT_TO_ANGLE_FACTOR);
    }
  }
  if (Serial.available() > 0) {
    inputByte = Serial.read();
    if (inputByte == '\n') {
      bufferReady = true;
      bufferIndex = 0;
    } else {
      if (bufferIndex < bufferSize) {
        inputBuffer[bufferIndex] = inputByte;
        bufferIndex++;
      }
    }
  }
}