#face cascade to support threaded operation
import cv2
from threading import Thread
import time
	
class FaceProfileCascade:
        
	def __init__(self, dummyframe):
 
		# initialize the frame 
		# dummy variables are used to set up the right internal data structure for tuples
		# include the stopped variable used to test if the thread should be stopped

		self.profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')		
		self.stopped = False
		self.frame=dummyframe
		self.rects=[]
		self.proctime=0
	
		self.readflag=0
	
		# start the thread
		print("FaceProfileCascade Thread starting")
		t1=Thread(target=self.run, args=()).start()
			
	# send a new frame to the thread
	def newFrame(self,new_frame):
		self.frame=new_frame              
		return True
        
        # set up the detector that loops around the latest frame in the thread memory
	def run(self):
		# keep looping infinitely until the thread is stopped

		framecount=0
		skip_count=100
		
		while True:
			start_time=time.time()
			
			#run cascade to identify faces
			faces = self.profile_cascade.detectMultiScale(self.frame, 1.1, 5)

			#reset rects
			self.rects=[]

			#set current rects
			for (x,y,w,h) in faces:
				self.rects.append((x, y, x+w, y+h))
					
			#identify that there is new data that has not been read
			self.readflag=1

			#calculate the processing time 
			self.proctime=time.time()-start_time
			#if framecount % skip_count==0:
			#	print("prof proc time: %f  count: %d" % (time.time()-start_time,framecount))
			#framecount+=1
                        
			if self.stopped:
				return
 
	# a function to return the array of rectangles that the detector found
	def get_rects(self):
		self.readflag=0
		return self.rects
		
	# a function to return the latest metric on the time it takes to process a frame
	def get_proctime(self):
		return self.proctime

	# a function to identify when there are new detection results
	def get_new_data_flag(self):
		# readflag=0 if the data has been read, 1 if it is new
		return self.readflag
		
	
	def stop(self):
		# indicate that the thread should be stopped
		print("FaceProfileCascade Thread stopping")
		self.stopped = True

