#include <Servo.h>5
#define ACTUATOR_1_PIN 3
#define ACTUATOR_2_PIN 5
#define ACTUATOR_3_PIN 6
#define ACTUATOR_4_PIN 9
#define ACTUATOR_5_PIN 10

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
  while(!Serial);
  
  //establishContact();
  
  // reserve 200 bytes for the inputString:
  //inputString.reserve(200);
}

void loop() {
  // Use the 'S' character to represent a servo movement with the following byte as the value in degrees
  if (bufferReady)
  {
    bufferIndex = 0;
    bufferReady = false;
    if (inputBuffer[0] == 'S')
    {
      servo_1.write(inputBuffer[1]);
      servo_2.write(inputBuffer[2]);
      servo_3.write(inputBuffer[3]);
      servo_4.write(inputBuffer[4]);
      servo_5.write(inputBuffer[5]);
    }
  } 
  if (Serial.available() > 0) {
    inputByte = Serial.read();
    if (inputByte == '\n')
    {
      bufferReady = true;
      bufferIndex = 0;
    }
    else
    {
      if (bufferIndex < bufferSize)
      {
        inputBuffer[bufferIndex] = inputByte;
        bufferIndex++;
      }
    }

}
}

/*
  SerialEvent occurs whenever a new data comes in the
 hardware serial RX.  This routine is run between each
 time loop() runs, so using delay inside loop can delay
 response.  Multiple bytes of data may be available.
 */
//void serialEvent() {
//  while (Serial.available()) {
////    // get the new byte:
////    char inChar = (char)Serial.read();
////    // add it to the inputString:
////    inputString += inChar;
////    // if the incoming character is a newline, set a flag
////    // so the main loop can do something about it:
////    if (inChar == '\n') {
////      stringComplete = true;
////    }
//    
//    inputByte[0] = Serial.read();
//    Serial.print("I received: ");
//    Serial.println(inputByte[0], DEC);
//  }
//}


