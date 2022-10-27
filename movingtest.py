#motion detection class to support threaded operation
import cv2
from threading import Thread
import time
import numpy as np
import imutils
from imutils import face_utils

from rex_CVFrameCapture2020 import FrameCapture  
	

stopped = False
detectionflag=0

myFrameCapture = FrameCapture()  #all capture paramters are in the json config file
frame=myFrameCapture.getFrame()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)


(height, width)=gray.shape
print ("size from gray image properties = ", width, height)

framecount=0
skip_count=10
#averageValue1 = np.float32(frame)  #initialze ave value with dummy frame
#averageValue1=frame
#moving_average = cv2.cv.CreateImage(cv2.GetSize(frame),32,3) # image to store running a
moving_average = np.float32(gray)

min_area=500

#initialize detection moving average at center
cx2=width/2
cy2=height/2


while True:
    start_time=time.time()

    frame=myFrameCapture.getFrame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)


    cv2.accumulateWeighted(gray, moving_average, 0.01); #//error in hereeeeee!
    resultingFrames = cv2.convertScaleAbs(moving_average)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(moving_average))


    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cx=0  #reset the sum accumulator to 0
    cy=0
    count=0


	# loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
        	continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        #ignore if near the edges
        if (x<20 or x>width-20 or y<20 or y>height-20):
            continue

        #draw contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
        cx=cx+(x+w/2)  #add to the sum of the centers
        cy=cy+ (y+h/2)
        count +=1      #increment the total
    
    if count>0:
        cx1=cx/count  #average = sum/count
        cy1=cy/count

    #create a moving average of detection centers
        alpha = 0.05
        cx2 = cx2*(1-alpha) + alpha*(cx1)
        cy2 = cy2*(1-alpha) + alpha*(cy1)
        #print (count, int(cx2), int(cy2))

    cv2.circle(frame, (int(cx2), int(cy2)) , 10, (0, 0, 255), -1)


    cv2.imshow('InputWindow', frame)


    #print (int((time.time()-start_time)*1000)
    #cv2.imshow('MovingAverage', resultingFrames)
    #cv2.imshow('Motion', frameDelta)
    #cv2.imshow('threshold', thresh)
    
    ##############  Process keyboard commands #######################
    # look for a 'q' keypress to exit
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

myFrameCapture.stop()
cv2.destroyAllWindows

