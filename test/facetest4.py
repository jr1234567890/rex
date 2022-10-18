#A simple program to test DLIB facial landmarks, with a software-based detector
#6/7/20

#Thanks go to Adrian at pyimagesearch for some fantastic tutorials and utilities

# import the necessary packages
import cv2
import numpy as np
import dlib
from time import sleep
import time

#include the image utilities from pyimagesearch.com
from imutils import face_utils

#initialize webcam 0
myframe = cv2.VideoCapture(0)
sleep(0.1)     #let the camera settle before sampling the data

#Use the dlib face detector
dlibdetector = dlib.get_frontal_face_detector()

#Use the dlib 5 landmark predictor
dlib_landmarks = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# setup the OpenCV frame to display on the screen in the upper left corner
winname = "Face Tracker"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 10, 30)  # Move it to the upper left corner

while(True):  
    start_time=time.time()

    #get new frame from the webcam
    (grabbed, frame) = myframe.read()


    #run dlib face dector on the frame, converted to grayscale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects=dlibdetector(gray)
    
    detect_time=time.time()
    
    #for each face found:
    for rect in rects:
        # compute the bounding box of the face and draw it on the frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (255, 255, 0), 1)

    	# use DLIB predictor to the the facial landmarks for the face
        print(rect)
        shape = dlib_landmarks(gray, rect)
        
        # convert the facial landmark (x, y)-coordinates to a NumPy array
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


