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
myPalmDetector = PalmCascade(gray)
myDetectFaces = DetectFaces(frame, proc_w, proc_h)
#myPlayAudio=PlayAudio()

#file for face landmark detection
#replaced 68 point detector with 5 point detector on 5/24/20
#landmarkpredictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
landmarkpredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=10, maxDistance=90)
trackers = []
trackableObjects = {}

# setup the frame to display on the screen in the upper left corner
winname = "Face Tracker"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 10, 30)  # Move it to as specific Window coordinate

#start_time = time.time()  #initialize the timer

# set up roar parameters and tracking parameters
roar_timer = time.time()
roar_flag = 0
roar_special_flag =0
roar_duration = 5.0   # the length of the roar audio, in seconds
roar_min_interval = 10  # the minimum time between starting roars
eye_cmd = 0            # the flag to turn on the LED eyes
mouth_pos = conf["mouth_closed"]
palms = 0
num_palms = 0
last_selected_object=999  #used to see if it is a new or old selection
new_object_flag=True       #used to identify when the selected object has changed
ID_object=999           #initialize the pointer to the selected face ID

ID_downtime=5   #the time between playing songs for any one person
ID_timer = [0]*1000  #set up a timer for each person
ID_unknown=True   

# initiate lists.  yes, I know I should not use static declarations - TODO - change to dynamic later
oldx = [250] * 10000
vx = [0.0] * 10000
xlist = [int(proc_w/2)] * 10000
ylist = [int(proc_h/2)] * 10000
start_x = [1] * 10000
end_x = [1] * 10000
start_y = [1] * 10000
end_y = [1] * 10000
area = [0] * 10000  
last_servo_dwell = [0] * 10000
num_hits = [0] * 10000
lasttime = [0.0] * 10000
interest = [0] * 10000        # the "is it interesting" metric
         # the area of each face


#TODO replace the above with a multidimensional array, and use dyanmic appends to create each one. 
#face_data[250,0.0,int(proc_w/2),int(proc_h/2),1,1,1,0.0,0,0]*10000
# 0   oldx
# 1   vx
# 2   xlist
# 3   ylist
# 4   start_x
# 5   end_x
# 6   start_y
# 7   end_y
# 8   last_servo_dwell
# 9   num_hits
# 10  lasttime
# 11  interest
# 12  area

# initialize servo target and other variables
midpoint = proc_w/2
loc = [midpoint, midpoint]
target = [midpoint, midpoint]

avg = None
previous_selection = 0
selected_object = 0

proc_time = 0.01
target_x = 0
target_y = 0
servo_x = proc_w/2
servo_y = 0

tilt_servo=90
mouth_opening=0

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

############         Initialize the face recognition function
if conf["enable_face_ID"]:
    embedding_model_name="recognizer/openface_nn4.small2.v1.t7"
    recognizername="recognizer/recognizer.pickle"
    le_name="recognizer/le.pickle"
    confidence_name=0.7

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model_name)
    if (platform.system()=="Linux"):  #see if it's Linux
            embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    #embedder.setPreferableTarget(cv2.dnn.DNN_BACKEND_OPENCV)
    print ("preferred target set up for face embedder")


    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(recognizername, "rb").read())
    le = pickle.loads(open(le_name, "rb").read())

#initialize the audio
wave_obj = sa.WaveObject.from_wave_file(conf["start_sound"])
play_obj = wave_obj.play()
#play_obj.stop()

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
    orig_frame==myFrameCapture.getFrame()
    fullframe=myFrameCapture.getFrameFull()
    scale=myFrameCapture.getScale()

    #cv2.imshow("orig frame", orig_frame)
    #cv2.imshow("working frame", frame)

    get_frametime=time.time()


    #############    PALM   DETECTOR   ###############
    #get halfsize grayscale frame from the frame grabber and send to the palm detector
    ##palmgray = cv2.cvtColor(myFrameCapture.getFrameSmall(), cv2.COLOR_BGR2GRAY)
    palmgray=myFrameCapture.getFrameSmall()
    myPalmDetector.newFrame(palmgray)

    # get the total number of palms detected in the frame
    if myPalmDetector.get_new_data_flag() == 1:  # check to see if there is a new detection
        palmrects = myPalmDetector.get_rects()

        # calculate a moving average for the number of palm detections
        alpha = 0.4
        palms = palms*(1-alpha) + alpha*(len(palmrects))
        num_palms = int(palms+.5)  # set to 2 if the moving average > 1.5

    #reset for each run through the loop
    x = 0
    x2 = 1
    y = 0
    y2 = 1
    rects = []
    newrects = []
    rects2 = []
    rects3 = []
    rects4 = []
    rects5 = []

    palmtime=time.time()

    #######    FACE DETECTOR #############

    #initialize a timer to measure face detection time
    detstart_time=time.time()

    #run the face detector
    rects=myDetectFaces.update(frame,conf["detection_confidence"])

    #calculate the face detection time
    detproctime=time.time()-detstart_time
    #dettime=time.time()
    
    #########   TRACK OBJECTS  ###########

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    
    objects = ct.update(rects)
    #Return structure: (ID, centroid), where centroid = (centroidX, centroidY, startX, startY, endX, endY)) 

    #DEBUG  print box around current rects in yellow
    #for i in range(0, len(rects)):
    #    tempstartx=rects[i][0]
    #    tempstarty=rects[i][1]
    #    tempendx=rects[i][2]
    #    tempendy=rects[i][3]
    #    cv2.rectangle(frame, (tempstartx-10, tempstarty-10), (tempendx+10, tempendy+10), (0,255,255), 1)

    # reset the current list of objects and build it fresh each cycle
    current_list = []
    #print ("                           ",objects.items())

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

        #draw a blue box around all detections, with 5 pixels added to each side
        #cv2.rectangle(frame, (start_x[objectID]-5, start_y[objectID]+5), (end_x[objectID]+5, end_y[objectID]+5), (255,0,0), 1)
    # print (objectID, start_x[objectID], start_y[objectID], end_x[objectID], end_y[objectID])

        ############## calculate the interest parameter   #################

        # start with num_hits
        # set to 0 if num_hits<4 (ignore intermittent detections)
        
        num_hits[objectID] += 1

        # increase the interest multiplier if the image is below the halfway point (point at kids)
        y_mult = 1
        if (ylist[objectID] > proc_h/2):
            y_mult = 1.2

        # ignore it until it has been seen a few times
        if num_hits[objectID] > 4:
            interest[objectID] = num_hits[objectID] * y_mult
        else:
            interest[objectID] = 0

    #######   Select the object of interest  ###################

    # servo selection rules
        # change the object every N seconds based on servo_dwell_time limit
        # reset the num_hits for that object to lower its interest, and reset the clock
        # also, determine if this is a new selected object

    if time.time()-servo_start_time > servo_dwell_time:
        # pick the highest interest object
        highest = 0
        for objectID in current_list:
            # if the interest for this object is the highest we've seen, select it
            if (interest[objectID] > highest):
                highest = interest[objectID]
                selected_object = objectID
        #print (selected_object)

        #set flag if the selected object is different than last time
        if (last_selected_object!=selected_object):
            new_object_flag=True
            last_selected_object=selected_object
        else:   
            new_object_flag=False
        #new_object_flag=True

        # reset the dwell timer and the hit counter for this object
        servo_start_time = time.time()
        num_hits[selected_object] = 0
        
    #servo_start_time = time.time()   # reset the dwell timer
    #   num_hits[selected_object] = 0   #reset the hit counter for this object

    # set the parameters for the selected target
    target_x = xlist[selected_object]
    target_y = ylist[selected_object]
    
    #if there are no detections, set everything to neutral 
    if len(current_list)==0:        
        #set the target to the neutral point
        target_x = proc_w/2
        target_y = proc_h/10  #set it to look up
        tilt_servo=90
        eye_angle=0
        ID_object=999   #set the recognizer ID to default
        #print ("no detections")

    else:  #else process the detection(s)

        ##############   head tilt ###############

        # create dlib rectangle for the selected object bounding box
        buffer=0  #make the box a little bigger than the DNN box
        buffery=0
        l=start_x[selected_object]
        t=start_y[selected_object]
        r=end_x[selected_object]
        b=end_y[selected_object]

        #get the image chip of the face
        faceBoxRectangle = dlib.rectangle(int(l/scale),int(t/scale),int(r/scale),int(b/scale))      
        (startX, startY, endX, endY) = (l,t,r,b)
        face12 = orig_frame[startY:endY, startX:endX]
        #cv2.imshow("target", face12)

        #get the image chip of the face in the full frame
        (startX, startY, endX, endY) = (int(l/scale),int(t/scale),int(r/scale),int(b/scale))
        face = fullframe[startY:endY, startX:endX]
        #cv2.imshow("full frametarget", face)
        #cv2.imshow("full frame", fullframe)
 
        ###################    Face Landmarks  ############################

        #run the dlib face landmark process, and covert dlib format back to numpy format
    
        #shape = landmarkpredictor(palmgray, faceBoxRectangle)
        shape = landmarkpredictor(fullframe, faceBoxRectangle)
        shape = face_utils.shape_to_np(shape)
        #cv2.imshow("full frame", fullframe)

        #DEBUG
        #Display landmarks
        #for (x,y) in shape:
        #    x=int(x*scale)
        #    y=int(y*scale)
        #    cv2.circle(frame, (x,y),4, (0,0,255),-1)

        #5/24/20  changed to 5 point landmakr from original 68 point detector
        #share array
        #0 = right outer
        #1 = right inner
        #2 = left outer
        #3 = left inner
        #4 = tip of nose
     
        #rightEyePts = shape[0:1]   #lStart:lEnd]
        #leftEyePts = shape[2:3]   #rStart:rEnd]
        #leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        #rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        
        #old 68 point detector, mapping to eye centerrs
        #create arrays of the points associated with eyes, and calculate the centroid
        # from Fig 2 of https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/


        leftEyePts = shape[37:42]   #lStart:lEnd]
        rightEyePts = shape[43:48]   #rStart:rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        #(p1,upperLip)=shape[63]  #upper lip point
        #(p2,lowerLip)=shape[67]  #lower lip
        #(p3, topNose)=shape[28]  #top of the nose
        #(p4, chin)=shape[9]  #bottom of chin

        #Mouth landmarks, inside of lips
        #           63
        #       62      64
        #   61              65
        #       68      66
        #           67

        # get xy coordinates of the mouth points
        #shape index is 0 to 67)
        (x61,y61)=shape[60]  
        (x62,y62)=shape[61]  
        (x63,y63)=shape[62]  
        (x64,y64)=shape[63]  
        (x65,y65)=shape[64]  
        (x66,y66)=shape[65]  
        (x67,y67)=shape[66]  
        (x68,y68)=shape[67]  


        #calculate the y axis mouth opening as a function of the mouth width
        mouth_opening=15*(abs(y68-y62) + abs(y67-y63) + abs(y66-y64))/( 2* abs(x65-x61))
        #print (int(mouth_opening))
        #Range is -1 to 12 for normal mouth movement. Up to 16 if I really gape open 
        #display a circle on the mouth with the size of the mouth opening
        circle_size=int(max(0,min(mouth_opening,20)))
        #print (circle_size)
        cv2.circle(frame, (int(x67*scale),int((y63+y67)/2*scale)),circle_size, (255,255,0),-1)
        #cv2.circle(frame, (x,y),4, (0,0,255),-1)

        # compute the angle between the eye centroids (arctan of deltaY/deltaX)
        # and smooth it out with a pseudo moving average
        eye_alpha=.8  # this is the moving average alpha - higher numbers make the change more gradual
        eye_angle_temp = np.degrees(np.arctan2((rightEyeCenter[1] - leftEyeCenter[1]) , (rightEyeCenter[0] - leftEyeCenter[0])))
        eye_angle =  eye_angle_temp * (1-eye_alpha) + eye_angle * eye_alpha

       
        #calculate servo command based on eye_angle multiplied by a config paramter
        tilt_servo= int(90 - conf["tilt_ratio"]*eye_angle)
        #print (tilt_servo)

        #compute the time it took to process the objects
        objectprocttime=time.time()-start_time
        objecttime=time.time()

    #################           FACE ID             #################

#TODO  - only do this until an ID is associated with an object, to prevent jitter
        #selected_object is the index to the selected target
        #new_object_flag is true if this is the first frame for a new object

        #logic:
        # if this is a new object, try to get an ID
        # if unknown, keep trying until you get it

        #toggle the ID_unknown flag to True if this is a different target

        if (new_object_flag==True):
            ID_unknown=True   

        
        if (conf["enable_face_ID"] and ID_unknown==True):
            print ("Attempting face id on target ", selected_object)
            facerecognitiontime=time.time()
            #define the box in the coordinates needed by the recognizer

            #(startX, startY, endX, endY) = (l,t,r,b)
            #create an image chip of the face
            #face = frame[startY:endY, startX:endX]
            #get the image height and width

            #10/26 - use the full frame chipout that was created in the landmark routine, above
            (fH, fW) = face.shape[:2]
            
            #Set the pointer to 999 as a default
            ID_object=999
            
            # ensure the face width and height are sufficiently large
            if fW > 20 and fH > 20:

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                    (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                #print("face ID embedder set up with blob")
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                #set the recognition ID if it meets the threshold
                if proba>conf["recognizer_threshold"]:
                    if (name=="Jeff"):  ID_object=1
                    if (name=="Kathy"): ID_object=2
                    if (name=="David"): ID_object=3
                    if (name=="Randy"): ID_object=4
                    if (name=="Ed"):    ID_object=5
                    if (name=="Rick"):  ID_object=6
                
                #if we got an ID, set the the ID_unknown flag to false
                if ID_object<999:
                    ID_unknown=False
                    new_object_flag=False

                # put the name and probablity on the frame
                # if proba>conf["recognizer_threshold"]:
                #     text = "{}: {:.2f}%".format(name, proba * 100)      
                #     cv2.putText(frame, text, (startX-20, endY+30),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),2)
                #     print (text)

        
            facerecognitiontime=time.time()-facerecognitiontime
    
    #faceIDtime=time.time()


    ##################### Set Servo target  ###########################

    # set the servo target using a pseudo moving average 
    #TODO investigate a PID instead of the moving average
    #alpha2 = 0.2  # this is the moving average alpha
    #servo_x = (1-alpha2)*servo_x + alpha2*target_x
    #servo_y = (1-alpha2)*servo_y + alpha2*target_y
    # 10/25/20  added a slewing function to arduino, no need for moving average calculation here
    servo_x=target_x  #left/right pointing
    servo_y=target_y   # up/down pointing

    ###########   Calculate servo pointing commands based on the target   #############

    # Convert servo_x and servo_y to servo command positions.

    # full range of the camera, after de-distortion
    x_fov = conf["camera_fov"]  # camera x field of view, in degrees
    y_fov = 90   # camera y field of view, in degrees

    # convert to center ref =  units are now -proc_w/2 to proc_w/2
    servo_x1 = -(servo_x-proc_w/2)
    # convert to angle based on FOV of the camera, from the center of the camera
    servo_x2 = servo_x1*x_fov/proc_w
    servo_x3 = servo_x2*sx_scale       # convert to servo angle
    pointx = int(servo_x3 + sx_center)  # add offset for the servo center position

    servo_y1=-(servo_y-proc_h/2)
    servo_y2=servo_y1*y_fov/proc_h
    servo_y3=servo_y2*sy_scale
    pointy=int(servo_y3+sy_center)
    #print(servo_y, servo_y1,servo_y2, servo_y3)

    #set Trex jaw servo to match mouth opening of the person.
    # mouth_open and mouth_closed are the servo min/max.
    # mouth_opening is the detected opening from the facial feature as a percentage of the face height.
    #   range is 0 to 12 since the servo range is about the same as the mouth opening range

    mo=conf["mouth_open"]      #nominally servo = 77
    mc=conf["mouth_closed"]    #nominally server = 90  
    mouth_pos= int (mc - mouth_opening)
    mouth_pos=max(mo,min(mouth_pos,mc))
    #print(mouth_pos)
    
    ###################    Manage the roar and audio play routine  #######################

    # set the roar flag if the number of hands is satisfied, and the inter-roar timer has been met
    # Feb 2020 - tried the PlayAudio function to launch a blocking audio player in its own thread
    # FEb 23 2020 - swithced to simpleaudio, since it is nonblocking and has a stop command
        

   #           ID_timer[ID_object]=time.time()

    #check to see if we have enough palms, the wait inteval has expired, and we are not currently roaring
    if((num_palms>=conf["hands_for_roar"]) and ((time.time()-roar_timer)> roar_min_interval) and (roar_flag==0)):
        #this is the initialization
        roar_flag=1
        print ("roar triggered with number of palms = ",  num_palms )
        roar_timer=time.time()  #start the clock
        #hands=3

        #play_obj.stop()  #this may not be necessary
        play_obj.stop()
        wave_obj = sa.WaveObject.from_wave_file(conf["trex_roar"])
        play_obj = wave_obj.play()       

        #myPlayAudio.launch(num_palms)  #launch the audio player based on the number of hands
    
    #check to see if we are in the middle of a roar
    if(roar_flag==1):
        #while roaring, overwriting the pointing commands 
        pointy=y_max                    # move the head to look up during the roar   
        #pointx=(x_max-x_min)/2    
        pointx=90
        mouth_pos=conf["mouth_open"]    # open the mouth
        eye_cmd=1                       # light up the eyes
        #print("roar flag")

        #if palms goes up during roar, override with special roar and set flag to only do it oncd
        if (roar_special_flag==0) and num_palms>=conf["hands_for_special_roar"] :
            print ("special roar triggered with hands = ", num_palms)
            play_obj.stop()
            wave_obj = sa.WaveObject.from_wave_file(conf["audio/trex_special_roar"])
            play_obj = wave_obj.play()  
            roar_special_flag=1

        #if roar time has expired reset servos and flags
        if((time.time()-roar_timer)> roar_duration):         
            print ("resetting roar")
            eye_cmd=0
            mouth_pos=conf["mouth_closed"]
            roar_flag=0   
            roar_special_flag=0 

    #if not in roar, see if we want to play a face-specific sound file
    else:
        # test to see if this is a new face ID
        if (ID_object!=last_selected_object):
            #test to see if we have responded to this ID recently
            if (time.time()-ID_timer[ID_object]) > ID_downtime  and ID_object<999:
                print("playing audio for face ID ", ID_object, time.time()-ID_timer[ID_object])
                if(ID_object==1): 
                    play_obj.stop()
                    wave_obj = sa.WaveObject.from_wave_file(conf["jeff_roar"])
                    play_obj = wave_obj.play() 
                if(ID_object==2): 
                    play_obj.stop()
                    wave_obj = sa.WaveObject.from_wave_file(conf["kathy_roar"])
                    play_obj = wave_obj.play() 
                if(ID_object==3): 
                    play_obj.stop()
                    wave_obj = sa.WaveObject.from_wave_file(conf["david_roar"])
                    play_obj = wave_obj.play() 
                if(ID_object==4): 
                    play_obj.stop()
                    wave_obj = sa.WaveObject.from_wave_file(conf["randy_roar"])
                    play_obj = wave_obj.play() 
                if(ID_object==5): 
                    play_obj.stop()
                    wave_obj = sa.WaveObject.from_wave_file(conf["ed_roar"])
                    play_obj = wave_obj.play()        
                if(ID_object==6): 
                    play_obj.stop()
                    wave_obj = sa.WaveObject.from_wave_file(conf["rick_roar"])
                    play_obj = wave_obj.play()        
            ID_timer[ID_object]=time.time()
                
    #define an oscillating offset to twiddle the x dimension to avoid an oscillation
    
    osc_time=.25
    osc_limit=5

    if time.time()-xtimer>osc_time:  #the time increment for an update
        #change the direction if the value is over the limit
        if osc_value>osc_limit:
            x_inc=-1
        if osc_value<-osc_limit:
            x_inc=1
        #update the oscillation value
        osc_value=osc_value+x_inc
        #reset the timer clock           
        xtimer=time.time() 

    #if pointing straight ahead, add the oscillation value to the x
    #TODO if(pointx==90):
    #    pointx=pointx+osc_value

    # write the command values to the arduino.

    # prevent pointx and pointy from exceeding the servo end stops
    if pointx<x_min: pointx=x_min
    if pointx>x_max: pointx=x_max
    if pointy<y_min: pointy=y_min
    if pointy>y_max: pointy=y_max
    if tilt_servo> 90+conf["max_tilt"]: tilt_servo=90+conf["max_tilt"]
    if tilt_servo< 90-conf["max_tilt"]: tilt_servo=90-conf["max_tilt"]

    #12/13/20  replace with numpy clip function to make sure servos don't exceed limits
    #1/10/21, np.clip is throwing an error.  put the manual if statements back in, above
    #pointx=np.clip(pointx,x_min,x+max)
    #pointy=np.clip(pointy,y_min,y_max)
    #tilt_servo=np.clip(tilt_servo,90-conf["max_tilt"], 90+conf["max_tilt"])

    #DEBUG
    #print  (pointx, pointy, mouth_pos, eye_cmd,tilt_servo,max_servo_slew)  
    #print  (pointx, pointy, mouth_pos, eye_cmd,tilt_servo)  


    servotime=time.time()
    if(skipflag==0):
     



        commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,tilt_servo)    
    #DEBUG
    #    print(pointx, pointy, mouth_pos, eye_cmd,tilt_servo,commandecho)

    servotime=(time.time()-servotime)*1
    ############    Create the Output Display  ################

    # calculate and display the output frames per second
    #totalFrames += 1
    frametime=1/(time.time()-looptime)
    #looptime=time.time()
    palmproctime=myPalmDetector.get_proctime()

    text =  "Frame (FPS)       {:03.0f}".format(frametime)
    text1 = "Face Detector(ms)  {:03.0f}".format(detproctime*1000)   
    text4 = "Palm Detector(ms)  {:03.0f} ".format(palmproctime*1000)  
    text5 = "Hands             {:03.1f}".format(num_palms)
    text6=  "Eye angle         {:03.1f}".format(eye_angle)
    text7=  "Face ID time (ms)  {:03.0f}".format(facerecognitiontime*1000)
    framemetric= myFrameCapture.getCaptureTime()
    if (framemetric==0):
        framemetric==999
    text8=  "Frame Capture (Hz)  {:03.0f}".format(int(1/(framemetric+0.001)))
    #print (commandecho[-4:])
    text9=  "Servo Current (A) "+ (commandecho[-4:]) #the last 4 characters is the current used by the servos

    # convert time stampes to individual durations
   
    servotime=servotime-faceIDtime
    #faceIDtime=faceIDtime-objecttime
    objecttime=objecttime-dettime
    dettime=dettime-palmtime
    palmtime=palmtime - get_frametime
    get_frametime=get_frametime-start_time
    loop=time.time()-looptime
    #reset loop timer
    looptime=time.time()
  
    text10 = "get frame {:03.0f}".format(get_frametime*100000)
    text11 = "palm      {:03.0f}".format(palmtime*100000)
    #text14 = "ID        {:03.0f}".format(faceIDtime*1000)
    text15 = "servo time     {:03.0f}".format(servotime*1000)
    #text16 = "loop      {:03.0f}".format(loop*1000)
    

    cv2.putText(frame, text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
    cv2.putText(frame, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)        
    cv2.putText(frame, text4, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text5, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text6, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text7, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text8, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text9, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text10, (300, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text11, (300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    #cv2.putText(frame, text13, (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text14, (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text15, (300, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    #cv2.putText(frame, text16, (300, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if (num_palms>=2):
        text = "HANDS {:03.1f}".format(num_palms)
        cv2.putText(frame, text, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # display the servo location 
    #text = "Servo x {:03.0f}, Servo y {:03.0f} Point x {:03.0f} Point y {:03.0f}".format(servo_x, servo_y, pointx, pointy)
    #cv2.putText(frame, text, (10, 30),
    #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    ########################       DRAW the detectinos, and the selected object  ################

    for (objectID, centroid) in objects.items():
        cx=xlist[objectID]
        cy=ylist[objectID]

        #draw the rectangle on the current frame
        cv2.rectangle(frame, (start_x[objectID], start_y[objectID]), (end_x[objectID], end_y[objectID]), (0,0,255), 2)

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        #cv2.circle(frame, (cx,cy), 4, (0, 255, 0), -1)

        # put the interest measurement next to the ID
        text = "interest {:05.0f}".format(interest[objectID])

        # make the objectID text black if low interest, and red if high interest
        if interest[objectID] < 200:  #black
            g1 = 0
            b1 = 255
            r1 = 255
        else:   #red
            g1 = 0
            b1 = 0
            r1 = 255

        cv2.putText(frame, text, (cx - 50, cy - -50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (g1, b1, r1), 2)

        # draw the area of all of the boxes
        #area[objectID]= (end_x[objectID]-start_x[objectID])*(end_y[objectID]-start_y[objectID])
        # this is not working as of 4/21.  I can't pass the area to the ObjectID list
        #text = "area {:05.0f}".format(area[objectID])
        #cv2.putText(frame, text, (cx- 50, cy - -50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the palm detections
    
    #print the servo target as a red cirle 
    cv2.circle(frame, (int(servo_x), int(servo_y)), 8, (0, 0, 255), -1)

    #draw detected palm boxes
    #as of 10/18/20, the palm frame is 1/2 the size of the regular frame
    for (x,y,x2,y2) in palmrects:
        cv2.rectangle(frame, (x*2, y*2), (x2*2, y2*2), (0,255,255), 2)

    # if there are detections, display the face box and other graphics
    if(len(current_list)>0):
        # show the selected box in a different color
        #cv2.rectangle(frame, (start_x[selected_object], start_y[selected_object]), (end_x[selected_object], end_y[selected_object]), (255,0,255), 1)

        #display the face landmarks on the selected face
        #for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        #display the line between the eyes
        thickness=4
        #print("drawing right eye ", rightEyeCenter, leftEyeCenter)
        #the raw points are in the fullframe scale, and have to be adjusted to the working frame size
        cv2.line(frame, (int(rightEyeCenter[0]*scale),int(rightEyeCenter[1]*scale)),(int(leftEyeCenter[0]*scale),int(leftEyeCenter[1]*scale)),(255,255,0) , 1) 
        
        #display the recognition ID
        if conf["enable_face_ID"]:
            if (ID_object==999): 
                name="Unknown"
                proba=0
            if (ID_object==1): name="Jeff"
            if (ID_object==2): name="Kathy"
            if (ID_object==3): name="David"
            if (ID_object==4): name="Randy"
            if (ID_object==5): name="Ed"
            if (ID_object==6): name="Rick"
            
            text = "{}: {:.2f}%".format(name, proba * 100)      
            cv2.putText(frame, text, (start_x[selected_object], start_y[selected_object]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
            #print (text)

    # display the servo scale data
    #text = "Servo: xy scale xy center {:03.1f} {:03.1f} {:03.0f} {:03.0f} ".format(sx_scale, sy_scale, sx_center, sy_center)
    #cv2.putText(frame, text, (10, 45),
    #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)    
            
    # display the image in the frame initialized above
    cv2.imshow(winname, frame)
    #cv2.imshow("fullframe", fullframe)
    #cv2.imshow("smallframe",palmgray)

    if(conf["output_video"]):
        # Write the frame into the file 'output.avi'  needs to be in BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    
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
myPalmDetector.stop()
myFrameCapture.stop()
#myPlayAudio.stop()

#reset head to neutral position
# 10/24/20 - slew the servos to the neutral position instead of a single command
if skipflag==0:
    # step=20
    # for i in range(0, step):
    #     slewx=int(pointx+ i*(90-pointx)/step)
    #     slewy=int(pointy+ i*(90-pointy)/step)
    #     slewmouth=int(mouth_pos+ i*(90-mouth_pos)/step)
    #     slewtilt =int(tilt_servo+ i*(90-tilt_servo)/step)

    #     commandecho=myRexCommand.update(slewx, slewy, slewmouth, 0,slewtilt)    
    #     print("slew to neutral ", step, " ", commandecho) 
    #     time.sleep(2/step)

    #10/25/20 - added slew function in arduino, and we don't need the above slew anymore
    commandecho=myRexCommand.update(90, 90, 90, 0, 90)

    

if(conf["output_video"]):
    sleep(0.5) 
    #stop the output file writing
    print ("Closing output file")
    out.release()
