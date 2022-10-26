# Using face recognition to point T-Rex head control servos at people in the crowd
# Copyright 2020 Jeff Reuter

#Updated to use Intel DNN processing stick, and to move screen capture to a separate thread
 
# Requires use of arduino running: rex2020r2.ino
#
# assumes the existence of file conf.json

# import the necessary packages

import sys
import cv2
import serial  # needed for pyserial interface to arduino
from time import sleep
from sys import exit
import dlib
import keyboard

from rex_ArduinoCommands import RexCommand     

myRexCommand=RexCommand()    #initialize the RexCommand function

value=90

pointx=90
pointy=90
mouth_pos=90
eye_cmd=1
tilt_servo=90

winname = "Face Tracker"
cv2.namedWindow(winname)  
temp=1

while(True):  # replace with some kind of test to see if WebcamStream is still active?

    #set the item of interest = value (which is adjusted by a and z keys)

    tilt_servo=value
    #pointx=int(90+temp)
    #if (temp==1):
    #        temp=0
    #else
    #    temp==1
    #temp=temp*-1
    #print (pointx)
    #tilt_servo=value
    #print(pointx)

    #print  (pointx, pointy, mouth_pos, eye_cmd,tilt_servo)  
    
    commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,tilt_servo)        
    print(commandecho)

     
        ##############  Process keyboard commands #######################
    # look for a 'q' keypress to exit
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
    #if keyboard.is_pressed('a'):
    if key == ord("a"):
        value=value+1
    if key == ord("z"):
        value=value-1

    sleep(0.1)

    ##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")


commandecho=myRexCommand.update(90, 90, 90, 0, 90)

cv2.destroyAllWindows

