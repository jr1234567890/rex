#This class pulls video frames from the camera, scales, crops and processes fisheye compensation and provides the latest frame on demand
#This also sends frames to the cascade threads so they always have the latest


#This is meant to run in a separate thread, to provide more CPU power to the main thread

#import cv2
from threading import Thread
import time
import json
from time import sleep
from sys import exit
import serial

# imports specific to centroid tracker
import numpy as np
import dlib
import platform   # for platform.system() to get the OS name


# import ComArduinoFunctions
# waitForArduino() Waits for startup and prints message
from ComArduinoFunctions import waitForArduino
# sendToArduino(sendStr)
from ComArduinoFunctions import sendToArduino  
# recvFromArduino()  Returns array of characters
from ComArduinoFunctions import recvFromArduino


class RexCommand:

#TODO - put in all the servo init and commands here

    def __init__(self):
        
        conf = json.load(open("conf.json"))
        self.skipflag=0
        self.waitingForReply=0


#this is from trex_test3.py
# #find com port with the text "Arduino" in the port description
# arduino_ports = [
#     p.device
#     for p in serial.tools.list_ports.comports()
# 	#for windows
#     #if 'COM' in p.description  # may need tweaking to match new arduinos
# 	#for linux
# 	if 'Arduino' in p.description  # may need tweaking to match new arduinos
# ]
# #print(arduino_ports)

# if not arduino_ports:
#     raise IOError("No Arduino found")
# if len(arduino_ports) > 1:
#     warnings.warn('Multiple Arduinos found - using the first')

# #connect to Arduino 	9200, 2880, 2840, 57600, 115200
# baudRate = 57600
# ser = serial.Serial(arduino_ports[0],baudRate)
# print ("Serial port " + arduino_ports[0] + " opened  Baudrate " + str(baudRate))

        if (platform.system()=="Linux"):  #see if it's Linux
        #if conf["linux"]==True:
            ser = serial.Serial("/dev/ttyACM0", conf["arduino_baud_rate"])
            print("Linux Serial port set up with baud rate:", conf["arduino_baud_rate"])

        else:  #else, set up the Windows com port     
            ser = serial.Serial(conf["win_arduino_port"],conf["arduino_baud_rate"])
            print ("Windows serial port " + conf["win_arduino_port"] + " opened  Baudrate ", conf["arduino_baud_rate"])

        self.ser=ser

        # Set up servo unit limits
        # These are determined by testing to be the extents of the servos before they hit a hard stop.
        # They may not be symmetric, but that will be adjusted with the center values below.
        self.x_min = conf["sx_min"]  # left`
        self.x_max = conf["sx_max"]  # right
        self.y_min = conf["sy_min"]  # down
        self.y_max = conf["sy_max"]  # up
      

    def update(self,x,y,jaw,eye,tilt,max_servo_slew):
         # write the servo value to the arduino.
        # data format for serial interface to arduino is "SERVO5" and 5 integers: 
            #x, y, mouth, eye on/off command and head tilt
        
        ser=self.ser

        arduino_string= "<SERVO5," + str(x) + "," + str(y) + "," + str(jaw) + "," + str(eye) + "," + str(tilt) + str(max_servo_slew) + ">"

        #skip the serial write if the arduino is not connected, skipflag==1
        if self.skipflag==0:
            if self.waitingForReply == False:
                sendToArduino(ser, arduino_string)
                # print ("Sent from PC - " + arduino_string )
                self.waitingForReply = True

            if self.waitingForReply == True:
                while ser.inWaiting() == 0:
                    pass  
                dataRecvd = recvFromArduino(ser)
                #print ("Reply Received  " , dataRecvd)
                self. waitingForReply = False
        else:  #else this is a test config without an arduino present, and we just wrap the input
            dataRecvd = arduino_string
      
        # return the echo from the arduino
        #print  (dataRecvd)
        return (dataRecvd)