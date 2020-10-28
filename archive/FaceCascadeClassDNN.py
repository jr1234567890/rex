#face cascade to support threaded operation
import cv2
from threading import Thread
import time
import numpy as np
    
class FaceCascade:
        
    def __init__(self, dummyframe):
        # initialize the frame 
        # dummy variables are used to set up the right internal data structure for tuples
        # include the stopped variable used to test if the thread should be stopped

        #print("test")

            
        prototxt = 'deploy.prototxt'
        model ='res10_300x300_ssd_iter_140000.caffemodel'
        confidence_limit=conf["detection_confidence"]
        #prototxt = 'MobileNetSSD_deploy.prototxt.txt'  #the full model
        #model ='MobileNetSSD_deploy.caffemodel'

        #self.net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
        self.net = cv2.dnn.readNetFromCaffe(prototxt,model)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        #self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #self.face_cascade.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRAID)
        #self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        #self.face_cascade = cv2.CascadeClassifier('palm.xml')
        #elf.face_cascade = cv2.CascadeClassifier('agest.xml')
        
        self.stopped = False
        self.frame=dummyframe
        self.rects=[]
        self.proctime=0
    
        self.readflag=0
    
        # start the thread
        print("FaceCascade Thread starting")
        t1=Thread(target=self.run, args=()).start()
            
    # send a new frame to the thread
    def newFrame(self,new_frame):
        self.frame=new_frame              
        return True
        
    # set up the detector that loops around the latest frame in the thread memory
    def run(self):
        # keep looping infinitely until the thread is stopped

        framecount=0
        skip_count=10
        
        while True:
            start_time=time.time()
            confidence=0
            #minSize = cv.Size(1, 1)
            #maxSize = cv.Size(90, 90)
            #gray, faces, 1.1, 3, 0, msize, msize
            #run cascade to identify faces
            #faces = self.face_cascade.detectMultiScale(self.frame, 1.1, 10, minSize, maxSize)  # orig 1.1, 5
            #faces = self.face_cascade.detectMultiScale(self.frame, 1.1, 10)  # orig 1.1, 5

            image=self.frame
            #DEBUG cv2.imshow("thread", image)

            confidence_limit=0.99

            (h,w)=image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300,300),(104.0,177.0,123.0))
            #blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300,300),127.5)
            
            self.net.setInput(blob)
            detections=self.net.forward()
            
            #2nd parameter = scaleFactor - 1.1 is a 10% size reduction per cascade  \
            #   smaller numbers are more accurate but more expensive computationally
            #3rd parameter - minNeighbors - higher number is higher quality, fewer detections

            #reset rects
            self.rects=[]

            #set current rects
            #for (x,y,w,h) in faces:
            for i in range(0, detections.shape[2]):
                confidence=detections[0,0,i,2]
             
                if confidence<confidence_limit:
                    continue
                box=detections[0,0,i,3:7] * np.array([w, h, w, h])
                (startx,starty,endx,endy) = box.astype("int")
                
                
                self.rects.append((startx, starty, endx, endy))
                #self.rects.append((x, y, x+w, y+h))
            print(confidence) 

            #identify that there is new data that has not been read
            self.readflag=1

            #print the statistics every skip_count frames                                                      
            
            if framecount % skip_count==0:
                self.proctime= time.time()-start_time
                #print("face proc time: %f  count: %d" % (time.time()-start_time,framecount))
            framecount+=1
            
            if self.stopped:
                return
 
    # a function to return the array of rectangles that the detector found
    def get_rects(self):
        self.readflag=0
        return self.rects

    # a function to identify when there are new detection results
    def get_new_data_flag(self):
        # readflag=0 if the data has been read, 1 if it is new
        return self.readflag
        
    # a function to report the processing time
    def get_proctime(self):
        return self.proctime

    def stop(self):
        # indicate that the thread should be stopped
        print("FaceCascade Thread stopping")
        self.stopped = True

