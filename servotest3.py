#Program to test the TRex mechanisms.  12/12/20

import sys
import serial  # needed for pyserial interface to arduino
import os.path
import platform   # for platform.system() to get the OS name
import time
import json
from time import sleep
from sys import exit
import serial

from rex_ArduinoCommands import RexCommand                  #Sends servo commands to the Arduino using serial port

conf = json.load(open("servotest.json"))

#### Arduino setup , using RexCommand funcitons
if conf["arduino"]==True:
    skipflag=0                   #indicates that the arduino is present, and should be included
    myRexCommand=RexCommand()    #initialize the RexCommand function
else:
    skipflag=1   #indicates the arduino is not present, and all commands should be skipped

print("Setup complete")


# Command: x  y mouth, eye LED, tilt, servo slew rate
pointx=90
pointy=90
mouth_pos=90
eye_angle=90
eye_cmd=0



xmin=55  # nin ins 45, bu tlimiting it to avoid hitting servo wire
xmax=125  # max is 135
ymin=32
ymax=157
mouthopen=77
mouthclosed=90
tiltmin=25
tiltmax=153
#max_servo_slew=50

delay=3  # seconds between


starttime=time.time()
mytime=0

while(True):  

    sleep(1)
    mytime=time.time()-starttime
    if (mytime>3) : pointx=xmax
    if (mytime>6) : pointx=xmin
    if (mytime>9) : pointx=90
    if (mytime>12) : pointy=ymax
    if (mytime>15) : pointy=ymin
    if (mytime>18) : pointy=90
    if (mytime>21) : mouth_pos=mouthopen
    if (mytime>22) : mouth_pos=mouthclosed
    if (mytime>23) : eye_cmd=1
    if (mytime>24) : eye_cmd=0
    if (mytime>25) : eye_angle=tiltmin
    if (mytime>28) : eye_angle=tiltmax
    if (mytime>31) : eye_angle=90
    
    if (mytime>34) : starttime=time.time()

   
    ##################   Send to Arduino  #############################

    # skip this if the arduino skip flag is set

    #print((str(mytime)[:4]),pointx, pointy, mouth_pos, eye_cmd,eye_angle)  
    success=1
    if(skipflag==0):

        success=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,eye_angle)  
        print ("sending to comm function",pointx, pointy, mouth_pos, eye_cmd,eye_angle)  
        commandecho=myRexCommand.get_response()  #this may not be the absolute latest, due to comms processing time
        print ("recieved from comm function ", commandecho)
        #commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,eye_angle)    
        #print(tilt_servo) 
    #DEBUG
       # print("success, command echo", success, commandecho)

##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")
myRexCommand.stop()



