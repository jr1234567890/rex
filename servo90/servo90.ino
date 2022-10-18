
#include <Servo.h>

Servo MyServo1;   //horizonal
Servo MyServo2;   //vertical
Servo MyServo3;   //mouth
Servo MyServo4;  //tilt

//Assign pins - select from HW PWM pins:  3,5,6,9,10,11
byte servoPin1 = 9;   //9
byte servoPin2 = 10;  //10
byte servoPin3 = 11;
byte servoPin4 = 6;
byte eyes=8;


void setup() {
  Serial.begin(57600);  //19200, 28800, 38400, 57600, 115200

  // initialize the servos
  MyServo1.attach(servoPin1);
  MyServo2.attach(servoPin2);
  MyServo3.attach(servoPin3);
  MyServo4.attach(servoPin4);

//  pinMode(A0, INPUT);
  pinMode(eyes, OUTPUT);
      
  MyServo1.write(90);  
  MyServo2.write(90);  
  MyServo3.write(90);  
  MyServo4.write(90);  
    
  digitalWrite(eyes, HIGH);

  // tell the PC we are ready
  Serial.println("<all servos set to 90");
}

//=============

void loop() {

  delay(1000);  //delay 5 ms, this will enable the slew function to work at a set interval
}

