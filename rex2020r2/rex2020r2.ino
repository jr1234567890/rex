/*  Program to receive servo inputs from the serial port and set servos
 *   Jeff Reuter 12/6/2020
 *      
 *   The first 3 message parameters are meant to be servos, centered at 90 deg
 *      horizontal
 *      vertical
 *      mouth
 *   The 4th parameter controls the LED eyes (on/off)
 *   The 5th parameter is the head tilt command
 *   The 6th paramter defines the max servo slew rate in deg/sec
 *   
 *   The message response is
 *      horizontal
 *      vertical
 *      mouth
 *      LED 
 *      tilt
 *      current
 *      average current
 *       
 *       
 *   Update for rex2020r2. Dec 2020    
 *    Added a max slew to the transmitted parameters
 *    Replaced pseudo moving average with max slew cap for horizonal, vertical and tilt
 *    Left mouth as instantaneous
 *       
 *   Update for rex2020r1    
 *    Added a slew to the selected servo values for horiz, vertical, and tilt
 *    Left mouth as instantaneous
 *       
 *   Update for Rex 2020, May 2020
 *      Added tilt parameter
 *      Changed overcurrent recovery to blink lights for 5 sec and then reset the program
 *      
 *   V14    
 *   Adds a current sensor, and sets all servos to 90 deg if the current exceeds a safety limit
 *       This is meant to avoid burning out the servos if commanded too far
 *       
 *   ToDo:
 *       put an if statement to reject if the first message field is not "SERVO5"
 *       Remove LED flashing, or modify it to be a heartbeat
 */

#include <Servo.h>
#include <avr/wdt.h>   //watchdog timer

Servo MyServo1;   //horizonal
Servo MyServo2;   //vertical
Servo MyServo3;   //mouth
Servo MyServo4;  //tilt

//Assign pins - select from HW PWM pins:  3,5,6,9,10,11
byte servoPin1 = 9;
byte servoPin2 = 10;
byte servoPin3 = 11;
byte servoPin4 = 6;

//assign the pins to trigger the audio player and light up the eyes
byte music_1=5;   //connect to trigger pin 1 of the player
byte music_2=6;   //connect to trigger pin 2 of the player
byte music_3=7;   //connect to trigger pin 3 of the player
byte eyes=8;

// define servo min/max to prevent overdrive
// this needs to be tweaked if servos arms are reinstalled
byte servo1Min = 41;
byte servo1Max = 127;
byte servo2Min = 70;
byte servo2Max = 145;
byte servo3Min = 72;
byte servo3Max = 94;
byte servo4Min = 70;
byte servo4Max = 110;


float max_current = 2.5; //exceeding this will trigger an abort
int overcurrent_state=0;
float overcurrent_duration=200;  //duration of current spike before disabling, milliseconds
float current=0.0;
float ave_current=0.0;
float alpha=0.9;
unsigned long overcurrent_timer;  //a timer to measure how long the current has spiked

// initialize servocommand to point to the center
int servo1=90;
int servo2=90;
int servo3=90;
int servo4=90;
int eye_cmd=0;

// initialize servo current positions
float Servo1Pos=90.0;
float Servo2Pos=90.0;
float Servo3Pos=90.0;
float Servo4Pos=90.0;

const byte buffSize = 40;
char inputBuffer[buffSize];
const char startMarker = '<';
const char endMarker = '>';
byte bytesRecvd = 0;
boolean readInProgress = false;
boolean newDataFromPC = false;

char messageFromPC[buffSize] = {0};
int newFlashInterval = 0;
float servoFraction = 0.0; // fraction of servo range to move

unsigned long lastTime;
unsigned long procTime;
unsigned long ms_timer;

int pin=0;
int pulse_length=125;  //pulse length to trigger FX audio player

//parameters for controlling slew rate of servos
float servorate=.02;  //this is the ratio of the movement we allow with each loop. 
int servodelay=5;  //milliseconds for the loop delay
float max_servo_slew=50.0;  // max servo motion rate, degreees/sec
float servo_step;

//=============

//this routine should reset the program to start over
//void softReset{
//asm volatile ("  jmp 0");
//}


void setup() {
  Serial.begin(57600);  //19200, 28800, 38400, 57600, 115200

  // initialize the servos
  MyServo1.attach(servoPin1);
  MyServo2.attach(servoPin2);
  MyServo3.attach(servoPin3);
  MyServo4.attach(servoPin4);

  pinMode(A0, INPUT);

  pinMode(eyes, OUTPUT);
  pinMode(13, OUTPUT);   //LED
    
  digitalWrite(eyes, LOW); // Start with the eyes off

  //blink LED twice to show it has started
   
  digitalWrite(13, HIGH);
  digitalWrite(eyes, HIGH);
  delay(100); 
  
  digitalWrite(13, LOW);
  digitalWrite(eyes, LOW);
  delay(100);
  
  digitalWrite(13, HIGH);
  digitalWrite(eyes, HIGH);
  
  delay(100); 
  digitalWrite(13, LOW);
  digitalWrite(eyes, LOW);

  //set servos to initial position
  updateServoPos();
  lastTime=micros();

  // tell the PC we are ready
  Serial.println("<Arduino is ready>");
}

//=============

void loop() {

  getDataFromPC();
  monitor_current();
  updateServoPos();  //includes turning eyes on and off   
  replyToPC();
  delay(servodelay);  //delay 5 ms, this will enable the slew function to work at a set interval
}

//=============

void getDataFromPC() {

    // receive data from PC and save it into inputBuffer
    
  if(Serial.available() > 0) {
    // receive a character from PC and save it into inputBuffer
    char x = Serial.read();

      // the order of these IF clauses is significant

    //look for the character that marks the end of the line.  ">"
    if (x == endMarker) {
      readInProgress = false;
      newDataFromPC = true;
      inputBuffer[bytesRecvd] = 0;
      parseData();
      procTime=micros()-lastTime;  // procTime= time since last endmarker read
      lastTime=micros();           //reset process timer
    }
    //if not the end of the line, get a new character and append it to the input buffer
    if(readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd ++;
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1;
      }
    }
    //if not a readInProgress, look for a start marker: "<" and then set the in process flag
    if (x == startMarker) { 
      bytesRecvd = 0; 
      readInProgress = true;
    }
  }
}

//=============
 
void parseData() {

    // split the data into its parts
    
  char * strtokIndx; // this is used by strtok() as an index
  
  strtokIndx = strtok(inputBuffer,",");      // get the first part - the string
  strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC
  
  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  servo1 = atoi(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  servo2 = atoi(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  servo3 = atoi(strtokIndx);     // convert this part to an integer
 
  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  eye_cmd = atoi(strtokIndx);     // convert this part to an integer
  
  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  servo4 = atoi(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  max_servo_slew = atoi(strtokIndx);     // convert this part to an integer
}

//=============

void replyToPC() {

  if (newDataFromPC) {
    newDataFromPC = false;
        
    Serial.print("<MsgFromArduino ");
    Serial.print(" ");
    Serial.print(int(Servo1Pos));
    Serial.print(" ");
    Serial.print(int(Servo2Pos));
    Serial.print(" ");
    Serial.print(servo3);
    Serial.print(" ");
    Serial.print(eye_cmd);
    Serial.print(" ");  
    Serial.print(servo4);
    Serial.print(" ");  
    Serial.print(current);
    Serial.print(" ");  
    Serial.print(ave_current);
    Serial.print(" ");  
    Serial.print(max_servo_slew);
    Serial.println(">");

  }
}

//=============

void updateServoPos() {
  //write servo pos if it has changed
    
    //10/25/20  added slew to the position to the target value with an exponential function
    //new position = position+ abs(servo_command - position) * slew rate per period   
    //current servo pos is a float, to allow for fractional closing to the commanded value
    //servos are approx .13 sec to go 60 deg, or roughly 375 deg/sec
    //the values of 5 ms delay with a .02 adjustment per period give a max rate of 80 deg/sec
    //see servo slew rate model.xlsx for the model

    //12/6/20  the slew was still pretty fast.  
    //This edit replaces it with a simple max slew rate to limit the speed of the trave.
    //servo1 is the value commanded from the PC, this is an int
    //Servo1Pos is the current commanded position.  this is a float to allow change to incrementally accrue
    //servodelay is the number of milliseconds between commands (nominally 5)
    //max_servo_slew is the max rate in degrees/sec

    /*
     * This is the pld moving average slew approach
    Servo1Pos=Servo1Pos+((float)servo1-Servo1Pos)*servorate;
    */

    //calculate servo increment
    float servo_step;
    servo_step=max_servo_slew*float(servodelay);
     
    //update the commanded position with the incremental movement
    //these are all floats, so incremental updates will eventually reach the command

//servo1

    //if the commanded position is greater than the current position 
    if (servo1-Servo1Pos>0.5) {
       Servo1Pos=Servo1Pos+servo_step;
    }
    //if the commanded position is less than the current position
    else if(servo1-Servo1Pos<-0.5) { 
       Servo1Pos=Servo1Pos-servo_step;
    }
    
    //else we are withing 0.5 degree, so just set it equal so we can skip servo writes we don't need
    else {
      Servo1Pos=float(servo1);
    }
    
    
    //write the new value if it is different than the original command
    if ((float)servo1!=Servo1Pos) {
      MyServo1.write(int(Servo1Pos));  
    } 

//servo2

   //if the commanded position is greater than the current position 
    if (servo2-Servo2Pos>0.5) {
       Servo2Pos=Servo2Pos+servo_step;
    }
    //if the commanded position is less than the current position
    else if(servo2-Servo2pos<-0.5) { 
       Servo2Pos=Servo2pos-servo_step;
    }
    
    //else we are withing 0.5 degree, so just set it equal so we can skip servo writes we don't need
    else {
      Servo2Pos=float(servo2);
    }
 
    if ((float)servo2!=Servo2Pos) {
      MyServo2.write(int(Servo2Pos));  
    } 
 
  if (servo3!=Servo3Pos){
      MyServo3.write(servo3);   
      Servo3Pos=servo3;
  }

    Servo4Pos=Servo4Pos+((float)servo4-Servo4Pos)*servorate;
    if (abs((float)servo4-Servo4Pos)<0.5) {
      Servo4Pos=(float)servo4;
    }
    if ((float)servo4!=Servo4Pos) {
      MyServo4.write(int(Servo4Pos));  
    } 
  

  if(eye_cmd==1){digitalWrite(eyes, HIGH);}
  else{digitalWrite (eyes, LOW); }
  
  }

void monitor_current() {
  // reads the current sensor and jumps to an "abort" configuration.
  //must reset the arduino to exit
  current=analogRead(A0);
  current = current * 5/1024;  //scale it to 5 amp full scale
  ave_current=ave_current*alpha + current*(1-alpha);

  if(ave_current<max_current){
    //if the state was 0 before, this is a no op
    //if the state was 1 before, this resets
    overcurrent_timer=millis();  //keep the timer current
    overcurrent_state=0;
  }
  else {
    if (overcurrent_state=0){    //if this is the first instance
      overcurrent_timer=millis();  //start the timer, number of ms since turn on
      overcurrent_state=1;         //set the state
    }
    if(millis()-overcurrent_timer>overcurrent_duration){//check the timer and run the abort sequence   
        Serial.println(" ");
        Serial.println(" ");
        Serial.print("<Msg ABORTED DUE TO OVERCURRENT.  Duration= ");
        Serial.print(millis()-overcurrent_timer);
        Serial.print("  Current = ");
        Serial.print(ave_current);
        Serial.println(">");
        Serial.println(" ");
        MyServo1.write(90);    
        MyServo2.write(90);    
        MyServo3.write(85);   
        MyServo4.write(90);   

        //blink the lights for 5 seconds
        ms_timer=millis();
        while((millis()-ms_timer)<5000) {
           digitalWrite (eyes, LOW);
           delay(250);
           digitalWrite (eyes, HIGH);
           delay(250);
        }  //end while

        //call the soft reset to restart the program
        //software_Reset() ;
        //enable the watchdog timer for 15 ms
        wdt_enable(WDTO_15MS);
        while(1){
          //loop until watchdog timer fires
        }

      } // end if timer
    }// end else
}
  

  
