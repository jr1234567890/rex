# trex servo test
# cycles through the min/max values of each servo

# Requires use of arduino code ArduinoPC2_servo13
#servo1 is pin 9,  x, left/right
#servo2 is pin 10, y, up/down
#servo3 is pin 11, z, the mouth
#eye is the LEDs in the eye

#keypress to adjust min and max values, with ascii mapping
# a 97 and s 115 adjust xmin
# d 100 and f 102 adjust xmax
# t 116 and g 103 adjust ymax
# h 104 and n 110 adjust ymin
# q to quit

import sys
import serial  # needed for pyserial interface to arduino
import msvcrt
import time
from time import sleep
import warnings
import serial.tools.list_ports

#import ComArduinoFunctions
from ComArduinoFunctions import waitForArduino  #Waits for startup and prints message
from ComArduinoFunctions import sendToArduino   #sends a string   
from ComArduinoFunctions import recvFromArduino #Returns array of characters

#set up serial interface to arduino
import warnings
import serial
import serial.tools.list_ports

#find com port with the text "Arduino" in the port description
arduino_ports = [
    p.device
    for p in serial.tools.list_ports.comports()
    if 'COM' in p.description  # may need tweaking to match new arduinos
]
if not arduino_ports:
    raise IOError("No Arduino found")
if len(arduino_ports) > 1:
    warnings.warn('Multiple Arduinos found - using the first')

#connect to Arduino 	9200, 2880, 2840, 57600, 115200
baudRate = 57600
ser = serial.Serial(arduino_ports[0],baudRate)
print ("Serial port " + arduino_ports[0] + " opened  Baudrate " + str(baudRate))

waitForArduino(ser) #wait for arduino to restart and be ready

waitingForReply=False  #set to False to trigger the send function first

#initial parameers  based on 4/27 test
# numbers reduced by 10 for torture test to reduce heat in the servos

max_x=  117  # 127
min_x=  51#  41
max_y=  135# 145
min_y=  80#  70
min_z=	77#72
max_z=  89#94
pointx=90
pointy=90
pointz=max_z
xcount=1  #will step up if positive, down if negative
ycount=1
xy_select=1  #1 means x is selected, -1 means y
eye=0     #the parameter to turn the eye on and off

#cycle through x and y min max sweeps to make a rectangle.
# enables the mouth servo and eye light at the end of each x cycle
#look for keypress at the end of an x and y sweep

while(1):
    # check to see if a key was pressed, get value, and adjust max/min
	if msvcrt.kbhit():
		char=ord(msvcrt.getch())
		print(char)
		if char==113: #q
			exit()

		# a 97 and s 115 adjust xmin
		elif char==97: 
			min_x=min_x+1
		elif char==115: 
			min_x=min_x-1

		# d 100 and f 102 adjust xmax		
		elif char==100:
			max_x=max_x-1
		elif char==102:
			max_x=max_x+1

		# t 116 and g 103 adjust ymax		
		elif char==116:
			max_y=max_y+1
		elif char==103: 
			max_y=max_y-1			

		# h 104 and n 110 adjust ymin
		elif char==104:
			min_y=min_y+1
		elif char==110: 
		
			min_y=min_y-1			
		#print (min_x, max_x, min_y, max_y)
	# cycle through square - x y min/max
	
	
	if (xy_select==1):  # step through an x
		pointx=pointx+xcount
		if ((pointx>max_x) or (pointx<min_x)): #if we hit the max or min  
			xcount=xcount*-1                 #change direction
			xy_select=xy_select*-1    		 #and shift to select the y cycle
			
			 #open/close the mouth and light the eyes on the transition to +
			if (xcount==1):
				pointz=max_z  
				eye=1;
			else :
				pointz=min_z			  #close the mount on the transition to -
				eye=0;			

	if (xy_select==-1):  # step through an y
		pointy=pointy+ycount
		if ((pointy>max_y) or (pointy<min_y)):        #if we hit the max or min  
			ycount=ycount*-1                 # change direction
			xy_select=xy_select*-1    		 # and shift to x 
			if (pointz==max_z): pointz=min_z
			elif(pointz==min_z): pointz=max_z
	
	#write the servo value to the arduino.
	#servo_1 is the y value (up and down)
	#servo_2 is the x value  (left and right)
	# data format for serial interface to arduino is "SERVO5" and 5 integers
	teststr= "<SERVO5," + str(pointx) + "," + str(pointy) + "," + str(pointz)+  ","+ str(eye) + ">"
	
	if waitingForReply == False:
		sendToArduino(ser, teststr)
		#print ("Sent from PC - " + teststr)
		waitingForReply = True

	if waitingForReply == True:
		while ser.inWaiting() == 0:
			pass  
		dataRecvd = recvFromArduino(ser)
		print ("Reply Received  " + dataRecvd)
		waitingForReply = False
	time.sleep(0.025)		
# cleanup
print ("shutting down")


