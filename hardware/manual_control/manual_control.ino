#include "VarSpeedServo.h"

#define ACTUATOR_1_PIN 3
#define ACTUATOR_2_PIN 5
#define ACTUATOR_3_PIN 6
#define ACTUATOR_4_PIN 9
#define ACTUATOR_5_PIN 10


#define INPUT_ACTUATOR_PIN_1 A1
#define INPUT_ACTUATOR_PIN_2 A2
#define INPUT_ACTUATOR_PIN_3 A3
#define INPUT_ACTUATOR_PIN_4 A4
#define INPUT_ACTUATOR_PIN_5 A5

#define HIGH_ANGLE 60
#define LOW_ANGLE 0

#define BIT_TO_ANGLE_FACTOR 180.0/255.0
#define MID_BIT 127.5


#define CONTROL_TYPE 0
//O FOR POSITION, 1 FOR SPEED

VarSpeedServo servo_1;
VarSpeedServo servo_2;
VarSpeedServo servo_3;
VarSpeedServo servo_4;
VarSpeedServo servo_5;


void setup() {
  servo_1.attach(ACTUATOR_1_PIN);
  servo_2.attach(ACTUATOR_2_PIN);
  servo_3.attach(ACTUATOR_3_PIN);
  servo_4.attach(ACTUATOR_4_PIN);
  servo_5.attach(ACTUATOR_5_PIN);

  // initialize serial:
  Serial.begin(115200);
}

void loop() {

    double value1 = analogRead(INPUT_ACTUATOR_PIN_1);
    double value2 = analogRead(INPUT_ACTUATOR_PIN_2);
    double value3 = analogRead(INPUT_ACTUATOR_PIN_3);
    double value4 = analogRead(INPUT_ACTUATOR_PIN_4);
    double value5 = analogRead(INPUT_ACTUATOR_PIN_5);

    if (CONTROL_TYPE == 1) { //speed control

      //between -255 and 255
      int decoded_speed1 = round(2*(value1-MID_BIT)); 
      int decoded_speed2 = round(2*(value2-MID_BIT));
      int decoded_speed3 = round(2*(value3-MID_BIT));
      int decoded_speed4 = round(2*(value4-MID_BIT));
      int decoded_speed5 = round(2*(value5-MID_BIT));

      //either 180 or 0
      int target_angle1 = (decoded_speed1 > 0) ? HIGH_ANGLE : LOW_ANGLE;
      int target_angle2 = (decoded_speed2 > 0) ? HIGH_ANGLE : LOW_ANGLE;
      int target_angle3 = (decoded_speed3 > 0) ? HIGH_ANGLE : LOW_ANGLE;
      int target_angle4 = (decoded_speed4 > 0) ? HIGH_ANGLE : LOW_ANGLE;
      int target_angle5 = (decoded_speed5 > 0) ? HIGH_ANGLE : LOW_ANGLE;

      servo_1.slowmove(target_angle1,abs(decoded_speed1));
      servo_2.slowmove(target_angle2,abs(decoded_speed2));
      servo_3.slowmove(target_angle3,abs(decoded_speed3));
      servo_4.slowmove(target_angle4,abs(decoded_speed4));
      servo_5.slowmove(target_angle5,abs(decoded_speed5));

    } else { //position control
      servo_1.write(value1*BIT_TO_ANGLE_FACTOR/180.0*(HIGH_ANGLE-LOW_ANGLE)+LOW_ANGLE);
      servo_2.write(value2*BIT_TO_ANGLE_FACTOR/180.0*(HIGH_ANGLE-LOW_ANGLE)+LOW_ANGLE);
      servo_3.write(value3*BIT_TO_ANGLE_FACTOR/180.0*(HIGH_ANGLE-LOW_ANGLE)+LOW_ANGLE);
      servo_4.write(value4*BIT_TO_ANGLE_FACTOR/180.0*(HIGH_ANGLE-LOW_ANGLE)+LOW_ANGLE);
      servo_5.write(value5*BIT_TO_ANGLE_FACTOR/180.0*(HIGH_ANGLE-LOW_ANGLE)+LOW_ANGLE);
    }
}
