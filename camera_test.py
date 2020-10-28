
import cv2
#import time
#import sys
#import json
#import numpy as np
#import imutils
#import platform   # for platform.system() to get the OS name


#USB Camera set/get indices
# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 5. CV_CAP_PROP_FPS Frame rate.
# 6. CV_CAP_PROP_FOURCC 4-character code of codec.
# 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

res=[640, 480]
#res=[1440,1080]

print ("Opening webcam")
#this is Windows
myframe = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
#(grabbed, frame) = myframe.read()


#cv2.CAP_DSHOW is a Windows option; added to get rid of black side bars on Arducam b0203 camera

#set the webcam resolution and framerate
myframe.set(3,res[0])   
myframe.set(4,res[1])    
myframe.set(5,25)      #framerate, nominally 25, but could be 29.97
#self.myframe.set(15, 0.1)  #exposure

#This works on the RPi webcam and the PC ball web cam
#myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

# this does not work on the Arducam
#Myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))



#test to see if the stream is open
if (myframe.isOpened()):
    print("FrameGrabber Thread started")     
else:   
    print("***************Framegrabber error: Failed to initialize  ***************")
    print ("           Check to see what webcam_ID is in the config file")
    print (" ")
    exit(0)

#DEBUG: #print camera parameters
for i in range (0,47):  
    print(i, myframe.get(i))


for x in range(0, 60):

    (grabbed, frame) = myframe.read()
    fullheight, fullwidth, channels = frame.shape  #this is the shape of the raw image
    
    print ("width, height, channels ", fullwidth, fullheight, channels, x)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('raw image', gray)

cv2.destroyAllWindows
