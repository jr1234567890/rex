#palm recognition cascade class to support threaded operation
import cv2
from threading import Thread
import time
	
class PalmCascade:
        
	def __init__(self, dummyframe):
		# initialize the frame 
		# dummy variables are used to set up the right internal data structure for tuples
		# include the stopped variable used to test if the thread should be stopped

		self.palm_cascade = cv2.CascadeClassifier('palm_v4.xml')

		#10/4/22  This fist cascade worked pretty well
		#self.palm_cascade = cv2.CascadeClassifier('fist_v3.xml')
				
		self.stopped = False
		self.frame=dummyframe
		self.rects=[]
		self.proctime=0
	
		self.readflag=0
	
		# start the thread
		print("Palm Cascade: Thread starting")
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
			#minSize = cv.Size(1, 1)
			#maxSize = cv.Size(90, 90)
			#gray, faces, 1.1, 3, 0, msize, msize
			#run cascade to identify faces
			#faces = self.face_cascade.detectMultiScale(self.frame, 1.1, 10, minSize, maxSize)  # orig 1.1, 5
			
			palms = self.palm_cascade.detectMultiScale(self.frame, 1.1, 10)  # orig 1.1, 5
			#print ("palm recognition", palms)
			
			#2nd parameter = scaleFactor - 1.1 is a 10% size reduction per cascade	\
			#	smaller numbers are more accurate but more expensive computationally
			#3rd parameter - minNeighbors - higher number is higher quality, fewer detections

			#reset rects
			self.rects=[]

			#set current rects
			for (x,y,w,h) in palms:
				self.rects.append((x, y, x+w, y+h))
				#print ("I see a palm in the thread")
					
			#identify that there is new data that has not been read
			self.readflag=1

			#print the statistics every skip_count frames                                                      
			
			if framecount % skip_count==0:
				self.proctime= time.time()-start_time
				#print("face proc time: %f  count: %d" % (time.time()-start_time,framecount))
			framecount+=1
			
			if self.stopped:
				print ("Palm Cascade Thread stopping")
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
		self.stopped = True

