#This function detects human bodies in an image frame.
#It uses the Intel processing stick if running on a Raspbery Pi

#This is meant to run in a separate thread, to provide more CPU power to the main thread

##Edit 10/15/21  - changed the Intel processing stick test to also test for ARM processor, and not just linux

import cv2
from threading import Thread
import time
#import argparse
#import warnings
import time
import json
from time import sleep
from sys import exit
import os

# imports specific to centroid tracker
from pyimagesearch.centroidtracker_jr import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import dlib
import platform   # for platform.system() to get the OS name

class DetectFaces:
    def __init__(self, frame, w, h):
        #self.frame2=frame
        # used by HOG
        #from imutils.object_detection import non_max_suppression
        #from fast_non_maximum_suppression import non_max_suppression_fast
        #from imutils import paths

        self.w=w
        self.h=h
            
        # set up the DNN recognizer
        prototxt = 'deploy.prototxt'
        #use this model with a blob of 300x300
        #model ='res10_300x300_ssd_iter_140000.caffemodel'
        #use this model with a blob of 800x600
        # from https://github.com/kgautam01/DNN-Based-Face-Detection
        model ='dnn_model.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(prototxt,model)
       
        #initialize the Intel processing stick as the target
        if (platform.system()=="Linux") and (os.uname()[4].startswith("arm")):  #see if it's the Rasp Pi - Linux running on an ARM
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
            print("Face Detector: initialized model for INTEL processing stick for Raspberry Pi")
        else:
            print("FaceDetection: initialized model in software")

    def update(self, frame, my_confidence):
        starttime=time.time()
        #set up the detector frame for processing
        #(h,w)=frame.shape[:2]
        #w=image_size[0]
        #h=image_size[1]
        #w=proc_w
        #scale=w/image_size[0]
        #h = int(h*scale-2*conf["y_crop"])
        #print("frame size 2", sys.getsizeof(testframe))

        # per a q&A on Adiran's post https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
        #Changing the blob values can get better results with smaller faces (further away)

        #original from his example
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300,300),(104.0,177.0,123.0))    
        
        #this worked, but did not detect small faces.
        #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (800, 600)), 1.0, (300,300),(104.0,177.0,123.0))   

        #tried this 10/15/21, based on https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
        #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (800, 600)), 1.0, (800,600),(104.0,177.0,123.0))   
        
        #modified per the comment
        #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 1.0, (400,400),(104.0,117.0,123.0))    
        self.net.setInput(blob)

        #run the detectcor
        detections=self.net.forward()

        #reset rects
        rects=[]

        #loop through the detections and create rect list

        w=self.w
        h=self.h
        self.confidence_limit=my_confidence
        #  

        #print(detections.shape[2])
        #TODO change this from detections.shape (which is 200 to a smaller number and see if performance improves)
        #results, 10/24/21, no discernable change when I changed this to 2 instead of 200
        for i in range(0, detections.shape[2]):
        #for i in range(0, 20):
            confidence=detections[0,0,i,2]
            if confidence<self.confidence_limit:
                continue
            box=detections[0,0,i,3:7] * np.array([w, h, w, h]) # creates box corners from 3,4,5,6. scaled up by original w and h
            (startx,starty,endx,endy) = box.astype("int") #converts to integers
 
            #add the rectangle to the rects structure
            rects.append((startx, starty, endx, endy))
 
        #print("frame processing time= ",(time.time()-starttime)*1000)

        # return the set of trackable objects
        return rects