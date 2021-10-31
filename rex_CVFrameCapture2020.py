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
import os


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
        res=conf["resolution"]        #the desired resolution of the camera
        self.xcrop=conf["x_crop"]     #the number of pixels to crop off the left and right
        self.ycrop=conf["y_crop"]     #the number of pixels to crop off the top and bottom
        #self.xscale=1.0
        #self.yscale=1.0
        self.fullheight=1
        self.fullwidth=1
        self.proc_h = 1
        self.scale=1.0
        self.framecapturerate=25  #framerate, nominally 25, but could be 29.97
        self.there_is_a_new_frame=True

        #print (res)

        #use test video if the config file parameter is true
        if (conf["use_test_video"]):
            #    for testing with a file
            self.myframe = cv2.VideoCapture('o.mp4')
            print ("Framegrabber: Opening o.mp4")
        else:
            print ("Framegrabber: Opening webcam")
           ##    for Linux
            #self.myframe = cv2.VideoCapture(0)  #src)  #"/dev/video0")

            ## for Windows, with Arducam b0203 camera
            #best settings seem to be to capture with 1920x1080 and scale down
            # if both x and y (index 3 and 4) are not correct, it will default to 640x480

            #use the platform.system() call to identify the OS< and use the appropriate CV capture routine
            print (platform.system())             

            if (platform.system()=="Linux"): 
                print (os.uname()[4])  #index 4 is machine type, such as ARM or x86_64
                if (os.uname()[4]=="x86_64"):
                    #this is the Linux PC 
                    print ("FrameGrabber: Starting video capture for Linux PC")
                    self.myframe = cv2.VideoCapture(-1)
                    self.myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

                else:
                    #this is the RPi
                    print ("FrameGrabber: Starting video capture for RPi")
                    self.myframe = cv2.VideoCapture(0)  #this works for RPi
                    self.myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
            else:
                #this is Windows
                print ("FrameGrabber: Starting video capture for Windows PC")
                self.myframe = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
                #cv2.CAP_DSHOW is a Windows option; added to get rid of black side bars on Arducam b0203 camera

            #set the webcam resolution and framerate
            self.myframe.set(3,res[0])   
            self.myframe.set(4,res[1])    
            self.myframe.set(5,self.framecapturerate)      #framerate, nominally 25, but could be 29.97
            #self.myframe.set(15, 0.1)  #exposure

            #This works on the RPi webcam and the PC ball web cam
            #self.myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

            # this does not work on the Arducam
            #self.myframe.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))

            #DEBUG: #print camera parameters
            #for i in range (0,47):  
            #    print(i, self.myframe.get(i))
        
        # ret, frame = self.myframe.read()
        # self.frame = self.frame[self.ycrop:self.y-self.ycrop, self.xcrop:self.x-self.xcrop]
#        self.frame = cv2.resize(self.frame, (self.proc_w, self.proc_h))
 #       self.frame = cv2.resize(self.frame, (self.proc_w, self.proc_h))
       
        #grab a frame so we can figure out the scales
        #(self.grabbed, self.frame) = self.myframe.read()
        #height, width, channels = self.frame.shape  #this is the shape of the raw image
        #self.y=height
        #self.x=width
        #print ("FrameGrabber: resolution from first camera query ", self.x, self.y)
        #fps=self.myframe.get(5)
        #print ("FrameGrabber initial image w, h, channels, FPS ", width, height, channels, fps)

        #self.scale=self.proc_w/width  #this is the overall scaling as we reduce to the proc_w width
        #self.proc_h = int(height*self.scale-2*self.ycrop)

        #print("FrameGrabber Initializing to w, h, scale", self.proc_w, self.proc_h, self.scale)

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

        #this is a dummy grab to end the setup.  It will be used to calculate frame sizes
        (self.grabbed, self.framefull) = self.myframe.read()
        self.fullheight, self.fullwidth, channels = self.framefull.shape  #this is the shape of the raw image
        self.framefull = self.framefull[self.ycrop:self.fullheight-self.ycrop, self.xcrop:self.fullwidth-self.xcrop]
        self.fullheight, self.fullwidth, channels = self.framefull.shape  #this is the shape of the cropped image
        print ("FrameGrabber Raw image after setting resolution and cropping ", self.fullwidth, self.fullheight)
        print ("FrameGrabber frame rate", self.myframe.get(5)) #index 5 is the FPS of the video stream myframe

        #calculate the desired height of the processing frame, keeping the same scale
        self.scale=self.proc_w/self.fullwidth
        self.proc_h=int(self.fullheight*self.scale)
        #print(self.proc_w, self.proc_h,self.scale)
        #self.proc_h=int(self.fullwidth*(self.proc_w/self.fullheight))

        # create the working frame
        self.frame = cv2.resize(self.framefull, (self.proc_w, self.proc_h))
        height, width, channels = self.frame.shape  #this is the shape of the  image
        print ("FrameGrabber resized image", width, height)

        #make an initial small grayframe to initialize the structure
        temp2 = cv2.resize(self.frame, (int(self.proc_w/2), int(self.proc_h/2)))
        #temp2 = cv2.resize(self.frame, (320,240))
        self.framesmall = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)

        #set the flag that there is a new frame available
        self.there_is_a_new_frame=True

       #self.xscale=width/self.fullwidth
       # self.yscale=height/self.fullheight
        #self.proc_h = int(height*self.scale-2*self.ycrop)
       
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
        
        time.sleep(5)  #let the initialization finish
        lastframetime = time.time()  #set up frame grabber timer

        print("Framegrabber: Frame update thread started")
    
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
        
            #sleep until it is time for a new frame

            #framerate=self.framecapturerate
            self.capturetime=time.time()-lastframetime
            currenttime=self.capturetime
            

            #if the framerate is too fast, this will slow it down
            sleeptime= .03-currenttime
            #print (sleeptime*1000)
            if (sleeptime>0):           
                time.sleep(sleeptime)
            #print((time.time()-lastframetime)*1000)
            lastframetime=time.time()
            #get the next frame

            (self.grabbed, self.framefull) = self.myframe.read()

            #crop it
            self.framefull = self.framefull[self.ycrop:self.fullheight-self.ycrop, self.xcrop:self.fullwidth-self.xcrop]

            #scale it
            #self.frame = cv2.resize(self.framefull, (500, 375))
            self.frame = cv2.resize(self.framefull, (int(self.proc_w), int(self.proc_h)))
            
            #make a small grayframe to speed up some recognizers
            temp2 = cv2.resize(self.frame, (int(self.proc_w/2), int(self.proc_h/2)))
            #temp2 = cv2.resize(self.frame, (320,240))
            self.framesmall = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)
     
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
        # return the frame most recently read and reset the "new frame" flag
        there_is_a_new_frame=False
        return self.frame

    def getFrameSmall(self):
        # return the small frame, for use by the hand recognizer
        return self.framesmall

    def getFrameFull(self):
        # return the original frame for use by landmarks and face recognizer
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

    def getScale(self):
        # return the scale from full size to working size frames
        return self.scale

    def getCaptureTime(self):
        # return the scale from full size to working size frames
        return self.capturetime

    def getNewFrameStatus(self):
        # return the scale from full size to working size frames
        return self.there_is_a_new_frame

    def stop(self):
        # indicate that the thread should be stopped
        print("FrameCapture thread stopping")
        self.stopped = True