
# import the opencv library 
import cv2 
import time
  
  
# define a video capture object 
vid = cv2.VideoCapture(-1) 
ret, frame = vid.read() 
#time.sleep(1)

#Notes:  the line below finally got the camera into the MJPG mode on LInux, to enable 25 Hz
#  see the text at the end of the file to show the available modes in the HDR camera.


vid.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
#vid.set(cv2.CAP_PROP_FOURCC ,cv2.CV_FOURCC('M', 'J', 'P', 'G') )
#vid.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'))
#vid.set(6,0)  #6 is the codec index, 0 is MJPG
#vid.set(6,1)  #6 is the codec index, 1 is YUYV

#set the webcam resolution and framerate
# NOTE: the 1920x1080 has the widest FOV, but gets a little distorted.
#res=[640,480]
#res=[800,600]
#res=[800,1064]
#res=[1920,1080]
res=[1280,720]



#print (res)
vid.set(3,res[0])   
vid.set(4,res[1])    
#vid.set(cv2.CAP_PROP_FPS , 5)  #cv2.CAP_PROP_FPS is 5
vid.set(5,25)      #framerate, nominally 25, but could be 29.97

ret, frame = vid.read() 
fullheight, fullwidth, channels = frame.shape  #this is the shape of the raw image  
print ("width, height, channels ", fullwidth, fullheight, channels)
print ("frame rate", vid.get(5))
print ("codec", vid.get(6))

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



#DEBUG: #print camera parameters
#for i in range (0,47):  
  #  print(i, vid.get(i))






  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    #print ("frame rate", vid.get(5))
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 


# results of testing high dynamic range usb camera
#           v4l2-ctl --list-formats-ext
# ioctl: VIDIOC_ENUM_FMT
# 	Type: Video Capture

# 	[0]: 'MJPG' (Motion-JPEG, compressed)
# 		Size: Discrete 576x648
# 			Interval: Discrete 0.040s (25.000 fps)
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 1920x1080
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 1280x720
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 800x1064
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 640x480
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 640x360
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 480x648
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 320x240
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 576x648
# 			Interval: Discrete 0.040s (25.000 fps)
# 			Interval: Discrete 0.040s (25.000 fps)
# 	[1]: 'YUYV' (YUYV 4:2:2)
# 		Size: Discrete 1920x1080
# 			Interval: Discrete 0.200s (5.000 fps)
# 			Interval: Discrete 0.200s (5.000 fps)
# 		Size: Discrete 1280x720
# 			Interval: Discrete 0.100s (10.000 fps)
# 			Interval: Discrete 0.200s (5.000 fps)
# 		Size: Discrete 800x1064
# 			Interval: Discrete 0.200s (5.000 fps)
# 		Size: Discrete 800x600
# 			Interval: Discrete 0.200s (5.000 fps)
# 		Size: Discrete 640x480
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 640x360
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 320x240
# 			Interval: Discrete 0.040s (25.000 fps)
# 		Size: Discrete 1920x1080
# 			Interval: Discrete 0.200s (5.000 fps)
# 			Interval: Discrete 0.200s (5.000 fps)

# v4l2-ctl --list-formats
# ioctl: VIDIOC_ENUM_FMT
# 	Type: Video Capture

# 	[0]: 'MJPG' (Motion-JPEG, compressed)
# 	[1]: 'YUYV' (YUYV 4:2:2)
# haunt@haunt:~/rex/test$ 
