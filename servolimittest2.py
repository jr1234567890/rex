# Using face recognition to point T-Rex head control servos at people in the crowd
# Copyright 2020 Jeff Reuter

#Updated to use Intel DNN processing stick, and to move screen capture to a separate thread
 
# Requires use of arduino running: rex2020r2.ino
#
# assumes the existence of file conf.json

# import the necessary packages

import sys
import serial  # needed for pyserial interface to arduino
from time import sleep
from sys import exit

from rex_ArduinoCommands import RexCommand     



myRexCommand=RexCommand()    #initialize the RexCommand function

value=90


while(True):  # replace with some kind of test to see if WebcamStream is still active?

    #set the item of interest = value (which is adjusted by a and z keys)
    pointx=value

    #print  (pointx, pointy, mouth_pos, eye_cmd,tilt_servo)  
    
    commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,tilt_servo,max_servo_slew)    
        
    print(commandecho)

     
        ##############  Process keyboard commands #######################
    # look for a 'q' keypress to exit
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("a"):
            value=value+1
    if key == ord("z"):
            value=value-1

    sleep(0.1)

    ##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")


commandecho=myRexCommand.update(90, 90, 90, 0, 90,1)

    

