#A simple program to test DLIB facial landmarks, with a software-based detector
#6/7/20

#Thanks go to Adrian at pyimagesearch for some fantastic tutorials and utilities

# import the necessary packages
import cv2
import numpy as np
import dlib
from time import sleep
import time

#image utilities for format conversions
from imutils import face_utils

#initialize webcam 0
myframe = cv2.VideoCapture(0)
sleep(0.1)     #let the camera settle before sampling the data

#Use the dlib face detector
#dlibdetector = dlib.get_frontal_face_detector()

# set up the DNN recognizer
prototxt = 'deploy.prototxt'
model ='res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt,model)

#initialize the Intel processing stick as the target
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#Use the dlib 5 landmark predictor
dlib_landmarks = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# setup the OpenCV frame to display on the screen in the upper left corner
winname = "Face Tracker"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 10, 30)  # Move it to the upper left corner

while(True):  
 

    #get new frame from the webcam
    (grabbed, frame) = myframe.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start_time=time.time()
    #create the blob needed for the DNN process
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 800)), 1.0, (300,300),(104.0,177.0,123.0))    
    #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 1.0, (400,400),(104.0,117.0,123.0))    
    net.setInput(blob)

    #run the detectcor
    detections=net.forward()

    #reset rects
    rects=[]

    #loop through the detections and create rect list
    (h,w)=frame.shape[:2]
    confidence_limit=.7

    for i in range(0, detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence<confidence_limit:
            continue
        box=detections[0,0,i,3:7] * np.array([w, h, w, h]) # creates box corners from 3,4,5,6. scaled up by original w and h
        #print(box)
        (startx,starty,endx,endy) = box.astype("int") #converts to integers

        #add the rectangle to the rects structure
        rects.append((startx, starty, endx, endy))
    
    detect_time=time.time()
       
    #for each face found:
    for rect in rects:
        # compute the bounding box of the face and draw it on the frame
        (l,t,r,b)=rect
        cv2.rectangle(frame, (l,t), (r,b), (255, 255, 0), 1)

        faceBoxRectangle = dlib.rectangle(l,t,r,b)
        print (faceBoxRectangle)

        #run the dlib face landmark process, and covert
        shape = dlib_landmarks(gray, faceBoxRectangle)
        shape = face_utils.shape_to_np(shape)
 
    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw each of them, with its ID number (i+1)
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
           
    landmark_time=time.time()-detect_time

    # display the image in the frame initialized above
    cv2.imshow(winname, frame)

    print("Detect Time (ms)", int((detect_time-start_time)*1000), "   Landmark Time (ms)", int(landmark_time*1000))

    ##############  Process keyboard commands #######################
    # look for keypress
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break


    ##########          End While(True) loop      ###############
    
# shutdown and cleanup
print ("shutting down")

# destroy the opencv video window
cv2.destroyAllWindows


