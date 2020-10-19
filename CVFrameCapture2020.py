#This class pulls video frames from the camera, scales, crops and processes fisheye compensation and provides the latest frame on demand
#This also sends frames to the cascade threads so they always have the latest


#This is meant to run in a separate thread, to provide more CPU power to the main thread

import cv2
from threading import Thread
import time
import sys
import json
#import numpy as np
import imutils
import platform   # for platform.system() to get the OS name


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

class FrameCapture:
    def __init__(self):
        # initialize the video camera stream and read the first frame
        # from the stream
        conf = json.load(open("conf.json"))

        self.xcrop=conf["x_crop"]
        self.ycrop=conf["y_crop"]
        self.flip=conf["180flip"]
        self.fisheye=conf["fisheye"]
        self.proc_w=conf["processing_width"]
        res=conf["resolution"]
        #print (res)

        #use test video if the config file parameter is true
        if (conf["use_test_video"]):
            #    for testing with a file
            self.myframe = cv2.VideoCapture('o.mp4')
            print ("Opening o.mp4")
        else:
            print ("Opening webcam")
           ##    for Linux
            #self.myframe = cv2.VideoCapture(0)  #src)  #"/dev/video0")

            ## for Windows, with Arducam b0203 camera
            #best settings seem to be to capture with 1920x1080 and scale down
            # if both x and y (index 3 and 4) are not correct, it will default to 640x480

            #use the platform.system() call to identify the OS< and use the appropriate CV capture routine
            print (platform.system())

            if (platform.system()=="Linux"):
                self.myframe = cv2.VideoCapture(0)
            else:
                self.myframe = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
                #cv2.CAP_DSHOW is a Windows option; added to get rid of black side bars on Arducam b0203 camera
            self.myframe.set(3,res[0])
            self.myframe.set(4,res[1])    
            self.myframe.set(5,25)   
            #self.myframe.set(15, 0.1)  #exposure
            self.myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

            # this does not work on the Arducam
            #self.myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))

        for i in range (0,47):
            print(i, self.myframe.get(i))
        
        # ret, frame = self.myframe.read()
        # self.frame = self.frame[self.ycrop:self.y-self.ycrop, self.xcrop:self.x-self.xcrop]
#        self.frame = cv2.resize(self.frame, (self.proc_w, self.proc_h))
 #       self.frame = cv2.resize(self.frame, (self.proc_w, self.proc_h))
       
        #grab a frame so we can figure out the scales
        (self.grabbed, self.frame) = self.myframe.read()
        height, width, channels = self.frame.shape  #this is the shape of the raw image
        self.y=height
        self.x=width
        #print ("frame resolution from camera query ", self.x, self.y)
        fps=self.myframe.get(5)
        print ("FrameGrabber Original image h, w, channels, FPS ", height, width, channels, fps)

        self.scale=self.proc_w/width  #this is the overall scaling as we reduce to the proc_w width
        self.proc_h = int(height*self.scale-2*self.ycrop)

        print("Initializing FrameGrabber to w, h, scale", self.proc_w, self.proc_h, self.scale)

        #self.frame = cv2.resize(frame, (self.proc_w, self.proc_h))

        #test to see if the stream is open
        if (self.myframe.isOpened()):
            self.stopped = False    
            t1=Thread(target=self.update, args=()).start()
            print("FrameGrabber Thread started")     
        else:   
            print("***************Framegrabber error: Failed to initialize  ***************")
            print ("           Check to see what webcam_ID is in the config file")
            print (" ")
            self.stopped=True
            return

        # enums for webcam get and set functions
        # 3 Width, 4 Height, 5 FPS, 6 codec , 10 bright, 11 contrast, 12 saturation, 13  hue, 14 gain, 15 exposure

        #this is a dummy grab to end the setup.  It will not be used
        (self.grabbed, self.frame) = self.myframe.read()
        #self.frame = self.frame[self.ycrop:self.y-self.ycrop, self.xcrop:self.x-self.xcrop]
        self.frame = cv2.resize(self.frame, (self.proc_w, self.proc_h))

        #cv2.imshow('raw image', self.frame)

        # #set the webcam resolution
        # xx=self.myframe.get(3)
        # yy=self.myframe.get(4)
        # self.fps=self.myframe.get(5)
        # print("FrameCapture Thread: video parameters: w:", xx,", h:",yy,", FPS:", self.fps)

            
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        
        #time.sleep(1)

        lastframetime = time.time()  #set up frame grabber timer

        print("Frame update thread started")
    
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
        
            #sleep until it is time for a new frame

            framerate=30
            currenttime=time.time()-lastframetime
            sleeptime= 1/framerate-currenttime

            if (sleeptime>0):           
                time.sleep(sleeptime)
            #get the next frame
            (self.grabbed, temp) = self.myframe.read()

            
            #crop it
            self.framefull = temp[self.ycrop:self.y-self.ycrop, self.xcrop:self.x-self.xcrop]

            #scale it
            self.frame = cv2.resize(self.framefull, (self.proc_w, self.proc_h))
            
            #make a small grayframe to speed up some recognizers
            temp2 = cv2.resize(self.frame, (int(self.proc_w/2), int(self.proc_h/2)))
            self.framesmall = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)

            lastframetime=time.time()
     
            # 180 deg rotation using cv2.flip
            if (self.flip):
                frame = cv2.flip(myframe, -1)

            
            # # apply fisheye undistort mapping
            #     if (self.fisheye):
            #         print("applying fisheye")
            #     # set up fisheye undistortion parameters
            #         if (self.x == 500):
            #             DIM = (500, 375)
            #             K = np.array([[141.5, 0.0, 249.1], [0.0, 142.0, 188.3], [0.0, 0.0, 1.0]])
            #             D = np.array([[-0.0396], [-0.00464], [-0.00150], [0.000310]])
            #         if self.x == 800:
            #             DIM = (800, 600)
            #             K = np.array([[225.0, 0.0, 398.0], [0.0, 225.0, 300.0], [0.0, 0.0, 1.0]])
            #             D = np.array([[-0.041], [-0.00086], [-0.0034], [0.00063]])
            #         if self.x == 1280:
            #             DIM = (1280, 1024)
            #             K = np.array([[359.0, 0.0, 637.0], [0.0, 358.0, 514.0], [0.0, 0.0, 1.0]])
            #             D = np.array([[-0.0330], [-0.0155], [-0.00637], [0.00143]])
            #         map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            #         frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
            #                                   borderMode=cv2.BORDER_CONSTANT)
            #     # this results in a frame that needs to be cropped.
                
            # crop off unusable sections of the top and bottom of the frame to make the detection faster
            #self.frame = self.frame[self.ycrop:self.y-self.ycrop, self.xcrop:self.x-self.xcrop]
            # resize to the desired processing witdth
            #print("orig size", sys.getsizeof(self.frame))
            #self.frame = cv2.resize(self.frame, (640,480))
            #self.frame = imutils.resize(self.frame, width="640")
            #print("after size", sys.getsizeof(self.frame))
           #print("numpy size", self.frame.shape)
    
    def getFrame(self):
        # return the frame most recently read
        #temp=sys.getsizeof(self.frame)
        #print("sending frame", temp)
        #print(self.myframe.shape)
        #print("Frame sent")
        return self.frame

    def getFrameSmall(self):
        # return the small frame, for use by the hand recognizer
        return self.framesmall

    def getFrameFull(self):
        # return the original frame
        return self.framefull
 
    def getSize(self):
        # return the frame size recently read
        x=self.myframe.get(3)
        y=self.myframe.get(4)
        #a=self.proc_w
        #b=self.proc_h
        mysize=[x, y]
        #print("sending size")
        return mysize

    def stop(self):
        # indicate that the thread should be stopped
        print("FrameCapture thread stopping")
        self.stopped = True