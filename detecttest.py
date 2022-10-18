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
from rex_FaceDetectionFunctions import DetectFaces      #The face detector, using HOG, and the Neural net stick


conf = json.load(open("conf.json"))

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

#Initialize the threads
myDetectFaces = DetectFaces(frame, proc_w, proc_h)

# setup the frame to display on the screen in the upper left corner
winname = "Face Detector"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 10, 30)  # Move it to as specific Window coordinate


start_x = [1] * 10000
end_x = [1] * 10000
start_y = [1] * 10000
end_y = [1] * 10000
xlist = [int(proc_w/2)] * 10000
ylist = [int(proc_h/2)] * 10000

start_time = time.time()  #initialize the timer

# set up roar parameters and tracking parameters

selected_object = 0

proc_time = 0.01

sleep(1)  #pause to let all the threads start up


##############   The main loop   ###############################

#initialize timers, measurements, and flags
starttime=time.time()
looptime=time.time()
facerecognitiontime=0
objectprocttime=0

ct = CentroidTracker(maxDisappeared=30, maxDistance=90)
trackers = []
trackableObjects = {}

while(True):  # replace with some kind of test to see if WebcamStream is still active?

    while(myFrameCapture.getNewFrameStatus==False):
        sleep(0.01)

    frame=myFrameCapture.getFrame()
    scale=myFrameCapture.getScale()

    #######    FACE DETECTOR #############

    #initialize a timer to measure face detection time
    detstart_time=time.time()

    #run the face detector
    rects=myDetectFaces.update(frame,conf["detection_confidence"])
    objects = ct.update(rects)
      
 # reset the current list of objects and build it fresh each cycle
    current_list = []

    # update trackable objects list and display
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current object ID
        # first, add it to the current_list
        current_list.append(objectID)

        #fetch the pointer to trackable object that matches this ID
        to = trackableObjects.get(objectID, None)
        
        # if the item is new in the object list, there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # get the current centroid position
        cx = int(centroid[0])
        cy = int(centroid[1])
        #startx_temp=int(centroid[2])

        # initialize a new object and then update the locations for each object
        if xlist[objectID] == None:  xlist.append(objectID)
        if ylist[objectID] == None:  ylist.append(objectID)
        if start_x[objectID] == None: start_x.append(objectID)
        if end_x[objectID] == None:   end_x.append(objectID)
        if start_y[objectID] == None: start_y.append(objectID)
        if end_y[objectID] == None:   end_y.append(objectID)
        
        #set the centroid and left, right, top, bottom values for the box
        xlist[objectID] = cx
        ylist[objectID] = cy
        start_x[objectID] = centroid[2]
        end_x[objectID] = centroid[4]
        start_y[objectID] = centroid[3]
        end_y[objectID] = centroid[5]


    # display detections
    for (objectID, centroid) in objects.items():
        #draw a blue box around all detections, with 5 pixels added to each side
        cv2.rectangle(frame, (start_x[objectID]-5, start_y[objectID]), (end_x[objectID]+5, end_y[objectID]), (255,0,0), 1)
    

    #################           FACE ID             #################


   

    ############    Create the Output Display  ################

    # calculate and display the output frames per second
    #totalFrames += 1
    frametime=1/(time.time()-start_time)
    start_time=time.time()
   

    text =  "Frame (FPS)       {:03.0f}".format(frametime)
    cv2.putText(frame, text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
   
            
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
myFrameCapture.stop()
#myPlayAudio.stop()
