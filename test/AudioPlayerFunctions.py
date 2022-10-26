#This class pulls video frames from the camera, scales, crops and processes fisheye compensation and provides the latest frame on demand
#This also sends frames to the cascade threads so they always have the latest


#This is meant to run in a separate thread, to provide more CPU power to the main thread


from threading import Thread

from playsound import playsound

from time import sleep

#import time
#import argparse
#import warnings
#import time
#import json
#from time import sleep
#from sys import exit

#
class PlayAudio:
    def __init__(self):
       
        self.stopped=False
		# start the thread
        print("Audio Thread starting")
        t1=Thread(target=self.run, args=()).start()

    def run(self):
        # keep looping infinitely until the thread is stopped
      		
        while True:
            sleep(1)	
            if self.stopped:
                return


    def launch(self, num_palms):    
        
        # start the music track based on the number of hands seen
        if (num_palms<3):
            print("starting roar audio")
            playsound('audio/t-rex_roar.mp3')                
        elif(num_palms<5):
            print("starting barney audio")
            playsound('audio/barney.wav')       
        else:
            print("starting frog audio")
            playsound('audio/frog.wav')  
        return


                
    def stop(self):
        # indicate that the thread should be stopped
        print("Audio Thread stopping")
        self.stopped = True
    