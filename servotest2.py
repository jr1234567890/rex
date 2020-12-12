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



xmin=41
xmax=145
ymin=70
ymax=145
mouthopen=77
mouthclosed=90
tiltmin=70
tiltmax=110
max_servo_slew=50


starttime=time.time()
mytime=0

while(True):  

#    sleep(0.004)
    mytime=time.time()-starttime
    if (mytime>2) : pointx=xmax
    if (mytime>4) : pointx=xmin
    if (mytime>6) : pointx=90
    if (mytime>8) : pointy=ymax
    if (mytime>10) : pointy=ymin
    if (mytime>12) : pointy=90
    if (mytime>14) : mouth_pos=mouthopen
    if (mytime>16) : mouth_pos=mouthclosed
    if (mytime>18) : eye_cmd=1
    if (mytime>20) : eye_cmd=0
    if (mytime>22) : eye_angle=tiltmin
    if (mytime>24) : eye_angle=tiltmax
    if (mytime>26) : eye_angle=90
    
    if (mytime>28) : starttime=time.time()

   
    ##################   Send to Arduino  #############################

    # skip this if the arduino skip flag is set

    #print((str(mytime)[:4]),pointx, pointy, mouth_pos, eye_cmd,eye_angle)  
    if(skipflag==0):
        commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,eye_angle)    
        #print(tilt_servo) 
    #DEBUG
        print(commandecho)

##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")



