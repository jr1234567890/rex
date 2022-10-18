# Using face recognition to point T-Rex head control servos at people in the crowd
# Copyright 2020 Jeff Reuter

#Updated to use Intel DNN processing stick, and to move screen capture to a separate thread
 
# Requires use of arduino running: rex2020r2.ino
#
# assumes the existence of file conf.json

# import the necessary packages
import imutils
import cv2
import sys
import serial  # needed for pyserial interface to arduino
import numpy as np
import os.path
import platform   # for platform.system() to get the OS name

from imutils import face_utils
import dlib
import pickle

import simpleaudio as sa

import time
import json
from time import sleep
from sys import exit

# imports specific to centroid tracker
from pyimagesearch.centroidtracker_jr import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import serial

# imports specific to threading
from threading import Thread

#import my functions
from rex_CVFrameCapture2020 import FrameCapture         # Captures video in a separate thread/core to speed up the main processing rate
from rex_PalmCascadeClass import PalmCascade            #Detects the number of palms
from rex_FaceDetectionFunctions import DetectFaces      #The face detector, using HOG, and the Neural net stick
from rex_ArduinoCommands import RexCommand                  #Sends servo commands to the Arduino using serial port

conf = json.load(open("conf.json"))

#### Arduino setup , using RexCommand funcitons
if conf["arduino"]==True:
    skipflag=0                   #indicates that the arduino is present, and should be included
    myRexCommand=RexCommand()    #initialize the RexCommand function
else:
    skipflag=1   #indicates the arduino is not present, and all commands should be skipped

# # Set up servo unit limits
# # These are determined by testing to be the extents of the servos before they hit a hard stop.
# # They may not be symmetric, but that will be adjusted with the center values below.
x_min = conf["sx_min"]  # left`
x_max = conf["sx_max"]  # right
y_min = conf["sy_min"]  # down
y_max = conf["sy_max"]  # up
max_servo_slew= conf["max_servo_slew"]

# set servo center and scale
# scale multiplies the angle from the camera to match the scale of the servo
#    a larger value will make the servo move MORE
# X/horizontal is nominally linear, and =1 since it is a direct drive in the mechanism
# Y/vertical is not linear due to sine relationship in the levered mechanism, but I'll start with that.
sx_scale = conf["sx_scale"]
sy_scale = conf["sy_scale"]
# center points define where the center of the video points to, in servo coords.  Nominally 90
sx_center = conf["sx_center"]
sy_center = conf["sy_center"]

print("Setup complete, starting image processing")

#############  Frame Capture setup **********************
myFrameCapture = FrameCapture()  #all capture paramters are in the json config file
sleep(3)     #let the camera settle before sampling the data

# get a frame from the stream function to initialize data & initialize the palm tracker
frame=myFrameCapture.getFrame()
orig_frame=myFrameCapture.getFrame()

proc_h, proc_w, channels = frame.shape  #this is the shape of the image coming out of the frame grabber
print ("Main program frame initialization: frame size =", proc_w, proc_h)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#output file setup
if(conf["output_video"]):
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('R','G','B','A'), 10, (proc_w,proc_h))
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (proc_w,proc_h))

#Initialize the threads
#myPalmDetector = PalmCascade(gray)
#myDetectFaces = DetectFaces(frame, proc_w, proc_h)
#myPlayAudio=PlayAudio()

#file for face landmark detection
#replaced 68 point detector with 5 point detector on 5/24/20
#landmarkpredictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
#landmarkpredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
#ct = CentroidTracker(maxDisappeared=30, maxDistance=90)
#trackers = []
#trackableObjects = {}

# setup the frame to display on the screen in the upper left corner
winname = "Test for Video Capture"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 10, 30)  # Move it to as specific Window coordinate

#start_time = time.time()  #initialize the timer

# set up roar parameters and tracking parameters
#roar_timer = time.time()

servo_start_time = 0.0   # the time to dwell on a target
servo_dwell_time = conf["face_dwell"]   # the time to dwell on a target before picking another
servo_ignore_time = 5.0

#frame_start_time = time.time()
#framerate = 30  # initialize the frame rate for averaging

# a timer to reset if there is nothing in view
i_see_nothing_flag=0  
i_see_nothing_activation_time=5 #seconds to wait to reaquire before resetting
i_see_nothing_sleep_time=10  #seconds to delay 
i_see_nothing_activation_timer=time.time()

# #initialize the pointing variables
pointx = 90
pointy = 90
eye_angle=0
#mypointx=90  #an extra x pointer for a side to side motion
xtimer=time.time()  #the timer for the side to side motion
osc_value=0 #an extra x pointer for a side to side motion
x_inc=1 # the increment to move the head, per time increment for side to side motin

sleep(1)  #pause to let all the threads start up


##############   The main loop   ###############################

#initialize timers, measurements, and flags
start_time=time.time()
looptime=time.time()
facerecognitiontime=0
objectprocttime=0
objecttime=0
dettime=0
palmtime=0
now=0
servotime=0
faceIDtime=0
get_frametime=0
debugtimer=0
loop=0
    

while(True):  # replace with some kind of test to see if WebcamStream is still active?
    start_time=time.time()

    #get new frame from the source
    #wait until a new frame is ready.
    while(myFrameCapture.getNewFrameStatus==False):
        sleep(0.01)

    frame=myFrameCapture.getFrame()
  

    servotime=time.time()
    ############    Create the Output Display  ################

    # calculate and display the output frames per second
    #totalFrames += 1
    #frametime=(time.time()-looptime)
    frametime=1/(time.time()-looptime)
    #print (frametime)
    #looptime=time.time()
    palmproctime=0
    detproctime=0
    num_palms=0
    eye_angle=0
    facerecognitiontime=0

    text =  "Frame (FPS)       {:07.2f}".format(frametime)
    #text1 = "Face Detector(ms)  {:03.0f}".format(detproctime*1000)   
    #text4 = "Palm Detector(ms)  {:03.0f} ".format(palmproctime*1000)  
    #text5 = "Hands             {:03.1f}".format(num_palms)
    #text6=  "Eye angle         {:03.1f}".format(eye_angle)
    #text7=  "Face ID time (ms)  {:03.0f}".format(facerecognitiontime*1000)
    framemetric= myFrameCapture.getCaptureTime()
    if (framemetric==0):
        framemetric==999
    text8=  "Frame Capture (Hz)  {:03.0f}".format(int(1/(framemetric+0.001)))

    # convert time stampes to individual durations
   
    #servotime=servotime-faceIDtime
    #faceIDtime=faceIDtime-objecttime
    #objecttime=objecttime-dettime
    #dettime=dettime-palmtime
    #palmtime=palmtime - get_frametime
    #get_frametime=get_frametime-start_time
    loop=time.time()-looptime
    #reset loop timer
    looptime=time.time()
  
    #text10 = "get frame {:03.0f}".format(get_frametime*1000)
    #text11 = "palm      {:03.0f}".format(palmtime*1000)
    #text12 = "det       {:03.0f}".format(dettime*1000)
    #text13 = "object    {:03.0f}".format(objecttime*1000)
    #text14 = "ID        {:03.0f}".format(faceIDtime*1000)
    #text15 = "servo     {:03.0f}".format(servotime*1000)
    text16 = "loop      {:03.0f}".format(loop*1000)
    

    cv2.putText(frame, text, (200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
    #cv2.putText(frame, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)        
    #cv2.putText(frame, text4, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text5, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text6, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text7, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text8, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    #cv2.putText(frame, text10, (300, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text11, (300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text12, (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text13, (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text14, (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text15, (300, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text16, (300, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    # display the image in the frame initialized above
    cv2.imshow(winname, frame)
    #cv2.imshow("fullframe", fullframe)
    #cv2.imshow("smallframe",palmgray)

   
    ############    End Output Display  ################        
    

    ##############  Process keyboard commands #######################
    # look for a 'q' keypress to exit
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break


    ##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")

# destroy the opencv video window
cv2.destroyAllWindows

# stop the threads
#myPalmDetector.stop()
myFrameCapture.stop()
#myPlayAudio.stop()

#reset head to neutral position
# 10/24/20 - slew the servos to the neutral position instead of a single command

