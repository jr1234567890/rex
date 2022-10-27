#motion detection class to support threaded operation
import cv2
from threading import Thread
import time
import numpy as np
import imutils
from imutils import face_utils
	
class MotionDetect:
        
	def __init__(self, dummyframe):
		# initialize the frame 
		# dummy variables are used to set up the right internal data structure for tuples
		# include the stopped variable used to test if the thread should be stopped
	
		self.stopped = False
		self.frame=dummyframe
		self.detectionflag=0
		self.detectioncenterx=0
		self.detectioncentery=0

		
		# start the thread
		print("MotionDetection: Thread starting")
		t1=Thread(target=self.run, args=()).start()

			
	# receive a new frame and send it to the thread
	def newFrame(self,new_frame):
		self.frame=new_frame              
		return True
        
	# set up the detector that loops around the latest frame in the thread memory
	def run(self):
		# keep looping infinitely until the thread is stopped

		min_area=500

		#create gray image to initialze structures from dummy frame
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		(height, width)=gray.shape
		print("Motion Detector started with size ", width,height)

		#initialize detection moving average at center
		cx2=width/2
		cy2=height/2

		#initialize moving average structure
		moving_average = np.float32(gray)

		while True:

			#get latest frame and preprocess blurred gray frame
			gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
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
				cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
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

				self.detectionflag=1
				self.detectioncenterx=int(cx2)
				self.detectioncentery=int(cy2)

			
			else:
				self.detectionflag=0
				self.detectioncenterx=0
				self.detectioncentery=0

			#cv2.circle(gray, (int(cx2), int(cy2)) , 10, (0, 0, 255), -1)
			cv2.imshow('InputWindow', gray)
		
			if self.stopped:
				print ("Motion Detect Thread stopping")
				cv2.destroyAllWindows()
				return
 
	def get_detection_flag(self):
		# readflag=0 if the data has been read, 1 if it is new
		return (self.detectionflag, self.detectioncenterx, self.detectioncentery)
		


	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

