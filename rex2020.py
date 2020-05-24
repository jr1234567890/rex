# Using face recognition to point T-Rex head control servos at people in the crowd
# Copyright 2020 Jeff Reuter

#Updated to use Intel DNN processing stick, and to move screen capture to a separate thread
 
# Requires use of arduino running: rex2020r0.ino
#
# assumes the existence of file conf.json

# import the necessary packages
import imutils
import cv2
import sys
import serial  # needed for pyserial interface to arduino
import numpy as np
import os.path

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
#import dlib
#import warnings
import serial

# import mp3/wav player
# from playsound import playsound

# imports specific to threading
from threading import Thread

#import my functions
from CVFrameCapture2020 import FrameCapture         # Captures video in a separate thread/core to speed up the main processing rate
from PalmCascadeClass import PalmCascade            #Detects the number of palms
from FaceDetectionFunctions import DetectFaces      #The face detector, using HOG, and the Neural net stick
from RexCommands import RexCommand                  #Sends servo commands to the Arduino using serial port
#rom AudioPlayerFunctions import PlayAudio          # a threaded audio player, based on # hands seen

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
sleep(0.1)     #let the camera settle before sampling the data

# get a frame from the stream function to initialize data & initialize the palm tracker
frame=myFrameCapture.getFrame()

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
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=30, maxDistance=90)
trackers = []
trackableObjects = {}

# setup the frame to display on the screen in the upper left corner
winname = "Face Tracker"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 10, 30)  # Move it to as specific Window coordinate

start_time = time.time()  #initialize the timer

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

############         Initialize the face recognition function
if conf["enable_face_ID"]:
    embedding_model_name="recognizer/openface_nn4.small2.v1.t7"
    recognizername="recognizer/recognizer.pickle"
    le_name="recognizer/le.pickle"
    confidence_name=0.7

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model_name)
    embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    #embedder.setPreferableTarget(cv2.dnn.DNN_BACKEND_OPENCV)
    print ("preferred target set up for face embedder")


    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(recognizername, "rb").read())
    le = pickle.loads(open(le_name, "rb").read())

#initialize the audio
wave_obj = sa.WaveObject.from_wave_file("Cat-purring-sound-4-sec.wav")
play_obj = wave_obj.play()
#play_obj.stop()


##############   The main loop   ###############################

#initialize timers, measurements, and flags
starttime=time.time()
looptime=time.time()
facerecognitiontime=0
objectprocttime=0



while(True):  # replace with some kind of test to see if WebcamStream is still active?

    # while(time.time()-looptime<(1/15)):
    #     sleep(0.001)
    # looptime=time.time()    

    #get new frame from the source
    frame=myFrameCapture.getFrame()
    #resize to 500 to speed up the process


    #############    PALM   DETECTOR   ###############
    #send grayscale frame to the palm detector
    palmgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    #######    FACE DETECTOR #############

    #initialize a timer to measure face detection time
    detstart_time=time.time()

    #run the face detector
    rects=myDetectFaces.update(frame)

    #calculate the face detection time
    detproctime=time.time()-detstart_time

    #########   TRACK OBJECTS  ###########

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    
    objects = ct.update(rects)
    #Return structure: (ID, centroid), where centroid = (centroidX, centroidY, startX, startY, endX, endY)) 

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
        # if (last_selected_object!=selected_object):
        #     new_object_flag=True
        #     last_selected_object=selected_object
        # else:   
        #     new_object_flag=False
        new_object_flag=True

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
        target_y = proc_h/2
        eye_angle=0
        ID_object=999   #set the recognizer ID to default
        print ("no detections")

    else:  #else process the detection(s)

        ##############  calculate head tilt for the selected object ###############

        # create dlib rectangle for the selected object bounding box
        l=int(start_x[selected_object])-10
        t=int(start_y[selected_object])
        r=int(end_x[selected_object])+10
        b=int(end_y[selected_object])
        faceBoxRectangle = dlib.rectangle(l,t,r,b)
        #print("face box", faceBoxRectangle)

        #run the dlib face landmark process, and covert dlib format back to numpy format
        shape = predictor(gray, faceBoxRectangle)
        shape = face_utils.shape_to_np(shape)

        #DEBUG
        #Display landmarks
        for (x,y) in shape:
            cv2.circle(frame, (x,y),1, (0,0,255),-1)

        #create arrays of the points associated with eyes, and calculate the centroid
        # from Fig 2 of https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        leftEyePts = shape[37:42]   #lStart:lEnd]
        rightEyePts = shape[43:48]   #rStart:rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids (arctan of deltaY/deltaX)
        # and smooth it out with a psuedo moving average
        eye_alpha=.5  # this is the moving average alpha
        eye_angle_temp = np.degrees(np.arctan2((rightEyeCenter[1] - leftEyeCenter[1]) , (rightEyeCenter[0] - leftEyeCenter[0])))
        eye_angle =  eye_angle_temp * (1-eye_alpha) + eye_angle * eye_alpha

       
        #calculate servo command based on eye_angle multiplied by a config paramter
        tilt_servo= int(90 + conf["tilt_ratio"]*eye_angle)
        # use numpy.clip to make sure the value is in the max tilt range
        tilt_servo=np.clip(tilt_servo,90-conf["max_tilt"], 90+conf["max_tilt"])

        #compute the time it took to process the objects
        objectprocttime=time.time()-starttime

    #################           Run face identifier on the selected face
        if (conf["enable_face_ID"] and new_object_flag==True):
            #define the box in the coordinates needed by the recognizer
            (startX, startY, endX, endY) = (l,t,r,b)
            #create an image chip of the face
            face = frame[startY:endY, startX:endX]
            #get the image height and width
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
                #sprint (ID_object)
  
                # put the name and probablity on the frame
                # if proba>conf["recognizer_threshold"]:
                #     text = "{}: {:.2f}%".format(name, proba * 100)      
                #     cv2.putText(frame, text, (startX-20, endY+30),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),2)
                #     print (text)

            new_object_flag=False
            facerecognitiontime=time.time()-starttime  - objectprocttime



    ##################### Set Servo target  ###########################

    # set the servo target using a pseudo moving average 
    #TODO investigate a PID instead of the moving average
    alpha2 = 0.2  # this is the moving average alpha
    servo_x = (1-alpha2)*servo_x + alpha2*target_x
    servo_y = (1-alpha2)*servo_y + alpha2*target_y

    ###########   Calculate servo pointing commands based on the target   #############

    # Convert servo_x and servo_y to servo command positions.

    # convert from pixels to servo angle commands
    # servo_x is in pixels, 0 to proc_w  (processing_width)
    # servo_y is in pixels, 0 (top) to proc_h (bottom)

    # full range of the camera, after de-distortion
    x_fov = conf["camera_fov"]  # camera x field of view, in degrees
    y_fov = 90   # camera y field of view, in degrees

    # servo_x and y, represented as an angle +/- from the center of the image
    # pointx and y represent the servo commands
    # servo_1 is the x value (up and down)
    # servo_2 is the y value  (left and right) - this is NOT quite a linear angle mapping
        # TODO - change this mapping to reflect the sine of angle, not the angle.

    # convert to center ref =  units are now -proc_w/2 to proc_w/2
    servo_x1 = -(servo_x-proc_w/2)
    # convert to angle based on FOV of the camera, from the center of the camera
    servo_x2 = servo_x1*x_fov/proc_w
    servo_x3 = servo_x2*sx_scale       # convert to servo angle
    pointx = int(-servo_x3 + sx_center)  # add offset for the servo center position

    servo_y1=-(servo_y-proc_h/2)
    servo_y2=servo_y1*y_fov/proc_h
    servo_y3=servo_y2*sy_scale
    pointy=int(servo_y3+sy_center)
    #print(servo_y, servo_y1,servo_y2, servo_y3)

    # prevent pointx and pointy from exceeding the servo end stops
    #TODO see if this single line should replace hte 2 lines
    #pointx=max(x_min,min(pointx,x_max))
    if pointx<x_min: pointx=x_min
    if pointx>x_max: pointx=x_max
    if pointy<y_min: pointy=y_min
    if pointy>y_max: pointy=y_max
    
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
        pointx=(x_max-x_min)/2    
        mouth_pos=conf["mouth_open"]    # open the mouth
        eye_cmd=1                       # light up the eyes

        #if palms goes up during roar, override with special roar and set flag to only do it oncd
        if (roar_special_flag==0) and num_palms>=conf["hands_for_special_roar"] :
            print ("special roar triggered with hands = ", num_palms)
            play_obj.stop()
            wave_obj = sa.WaveObject.from_wave_file(conf["trex_special_roar"])
            play_obj = wave_obj.play()  
            roar_special_flag=1
        
        # if (hands==3)and(num_palms>=5):
        #         myPlayAudio.launch(num_palms)  #launch the audio player based on the number of hands
        #         hands=5

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
            if (time.time()-ID_timer[ID_object]) > ID_downtime  :
                print("playing audio for ", ID_object, time.time()-ID_timer[ID_object])
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
                

    # write the command values to the arduino.
    if(skipflag==0):
        commandecho=myRexCommand.update(pointx, pointy, mouth_pos, eye_cmd,tilt_servo)     
    #DEBUG
        #print(commandecho)

    ############    Create the Output Display  ################

    # calculate and display the output frames per second
    #totalFrames += 1
    frametime=1/(time.time()-start_time)
    start_time=time.time()
    palmproctime=myPalmDetector.get_proctime()

    text =  "Frame (FPS)       {:05.2f}".format(frametime)
    text1 = "Face Detector(ms)  {:05.02f}".format(detproctime*1000)
    text4 = "Palm Detector(ms)  {:05.02f}".format(palmproctime*1000)
    text5 = "Hands             {:03.1f}".format(num_palms)
    text6=  "Eye angle         {:03.1f}".format(eye_angle)
    text7=  "Face ID time (ms)  {:05.02f}".format(facerecognitiontime*1000)

    cv2.putText(frame, text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
    cv2.putText(frame, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)        
    cv2.putText(frame, text4, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text5, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text6, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, text7, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if (num_palms>=2):
        text = "HANDS {:03.1f}".format(num_palms)
        cv2.putText(frame, text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # display the servo location 
    #text = "Servo x {:03.0f}, Servo y {:03.0f} Point x {:03.0f} Point y {:03.0f}".format(servo_x, servo_y, pointx, pointy)
    #cv2.putText(frame, text, (10, 30),
    #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for (objectID, centroid) in objects.items():
        cx=xlist[objectID]
        cy=ylist[objectID]

        #draw the rectangle on the current frame
        cv2.rectangle(frame, (start_x[objectID], start_y[objectID]), (end_x[objectID], end_y[objectID]), (0,0,255), 2)

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (cx - 20, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        #cv2.circle(frame, (cx,cy), 4, (0, 255, 0), -1)

        # put the interest measurement next to the ID
        text = "interest {:05.0f}".format(interest[objectID])

        # make the objectID text black if low interest, and red if high interest
        if interest[objectID] < 200:  #black
            g1 = 0
            b1 = 0
            r1 = 0
        else:   #red
            g1 = 0
            b1 = 0
            r1 = 255

        cv2.putText(frame, text, (cx - 50, cy - -30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (g1, b1, r1), 2)

        # draw the area of all of the boxes
        #area[objectID]= (end_x[objectID]-start_x[objectID])*(end_y[objectID]-start_y[objectID])
        # this is not working as of 4/21.  I can't pass the area to the ObjectID list
        #text = "area {:05.0f}".format(area[objectID])
        #cv2.putText(frame, text, (cx- 50, cy - -50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the palm detections
    
    #print the servo target as a red cirle 
    cv2.circle(frame, (int(servo_x), int(servo_y)), 8, (0, 0, 255), -1)

    #draw detected palm boxes
    for (x,y,x2,y2) in palmrects:
        cv2.rectangle(frame, (x, y), (x2, y2), (0,255,255), 2)

    # if there are detections, display the face box and other graphics
    if(len(current_list)>0):
        # show the selected box in a different color
        cv2.rectangle(frame, (start_x[selected_object], start_y[selected_object]), (end_x[selected_object], end_y[selected_object]), (255,0,255), 1)

        #display the face landmarks on the selected face
        #for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        #display the line between the eyes
        thickness=4
        #print("drawing right eye ", rightEyeCenter, leftEyeCenter)
        cv2.line(frame, (rightEyeCenter[0],rightEyeCenter[1]),(leftEyeCenter[0],leftEyeCenter[1]),(255,255,0) , thickness) 
        
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
            cv2.putText(frame, text, (start_x[selected_object], start_y[selected_object]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) 
            #print (text)





    # display the servo scale data
    #text = "Servo: xy scale xy center {:03.1f} {:03.1f} {:03.0f} {:03.0f} ".format(sx_scale, sy_scale, sx_center, sy_center)
    #cv2.putText(frame, text, (10, 45),
    #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)    
            
    # display the image in the frame initialized above
    cv2.imshow(winname, frame)

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
if skipflag==0:
    commandecho=myRexCommand.update(90, 90, 90, 0, 90)

if(conf["output_video"]):
    sleep(0.5) 
    #stop the output file writing
    print ("Closing output file")
    out.release()
