# Modifed version of rex2020.py to just test the servos.

#Updated to use Intel DNN processing stick, and to move screen capture to a separate thread
 
# Requires use of arduino running: rex2020r2.ino
#
# assumes the existence of file conf.json

# import the necessary packages
import sys
import serial  # needed for pyserial interface to arduino
import numpy as np
import os.path
import platform   # for platform.system() to get the OS name

import time
import json
from time import sleep
from sys import exit
import numpy as np
import serial

# imports specific to threading
#from threading import Thread

#import my functions
#from rex_CVFrameCapture2020 import FrameCapture         # Captures video in a separate thread/core to speed up the main processing rate
#from rex_PalmCascadeClass import PalmCascade            #Detects the number of palms
#from rex_FaceDetectionFunctions import DetectFaces      #The face detector, using HOG, and the Neural net stick
from rex_ArduinoCommands import RexCommand                  #Sends servo commands to the Arduino using serial port

conf = json.load(open("servotest.json"))

#### Arduino setup , using RexCommand funcitons
if conf["arduino"]==True:
    skipflag=0                   #indicates that the arduino is present, and should be included
    myRexCommand=RexCommand()    #initialize the RexCommand function
else:
    skipflag=1   #indicates the arduino is not present, and all commands should be skipped

# # Set up servo unit limits
# # These are determined by testing to be the extents of the servos before they hit a hard stop.
# # They may not be symmetric, but that will be adjusted with the center values below.
# Note that there are also min/max limits in the arduino code
x_min = conf["sx_min"]  # left`
x_max = conf["sx_max"]  # right
y_min = conf["sy_min"]  # down
y_max = conf["sy_max"]  # up

max_servo_slew= conf["max_servo_slew"]

# #initialize the pointing variables for the servo test
pointx = 90
pointy = 90
eye_angle=0
mouth_pos = conf["mouth_closed"]
eye_cmd = 0            # the flag to turn on the LED eyes

xinc=1
yinc=1
eyeinc=1
binary_timer=0

print("Setup complete")


while(True):  

    sleep(0.001)

    #increment servo settings
    pointx=pointx+xinc
    pointy=pointy+yinc
    exe_angle=eye_angle+eyeinc

    if pointx>x_max or pointx<x_min:
        xinc=-xinc

    if pointx>x_max or pointx<x_min:
        yinc=-yinc

    if eye_angle>30 or eye_angle<30:
        eyeinc=-eyeinc
      

    #increment the binary event timer and toggle settings if it exceeds the timer limit
    binary_timer = binary_timer+1
    if binary_timer>5000:
        if mouth_pos==conf["mouth_closed"]:
            mouth_pos=conf["mouth_open"]
        else:
            mouth_pos=conf["mouth_closed"]
        if eye_cmd==0:
            eye_cmd=1
        else :  
            eye_cmd=0
        
        binary_timer=0  #reset the timer

   
    ##################   Send to Arduino  #############################

    # skip this if the arduino skip flag is set
    print(pointx, pointy, mouth_pos, eye_cmd,eye_angle,max_servo_slew)  
    if(skipflag==0):
        commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,eye_angle,max_servo_slew)    
        #print(tilt_servo) 
    #DEBUG
        print(commandecho)

##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")

