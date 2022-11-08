from flask import Flask, render_template, Response
import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem
import Jetson.GPIO as GPIO
import signal

numCams = 6
led_pin=12
noStitchJustStack = True
Cylindrical = False
global cam

GPIO.setmode(GPIO.BOARD) 
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

print("Starting flask")
app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static')

def kill_signal_handler(signum, frame):
    print(f"Received kill signal {signum}. Exiting...")
    GPIO.output(led_pin, GPIO.LOW)
    sys.exit(0)
 
signal.signal(signal.SIGINT,kill_signal_handler)
signal.signal(signal.SIGTERM,kill_signal_handler)

def recalibrateCams():
   global cam
   print("Recalibrating cams")
   frames = cam.captureCameraImages()

   if numCams == 1:
      print("Num cams is 1, nothing to recalibrate")

   elif numCams == 3:
      Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2], save=True, filename="testFlaskHomography.npy")

   elif numCams == 6:
     if not Cylindrical:
         #Hl, Hr, Hl2, Hr2 = cam.calibrateMatrixTripleTwice(frames, save=True, filename="testFlaskHomography6.npy")
         Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2], save=True, filename="testFlaskHomography0.npy")
         Hl2, Hr2 = cam.calibrateMatrixTriple(frames[3], frames[4], frames[5], save=True, filename="testFlaskHomography1.npy")
     else:
         frames = cam.cylWarpFrames(frames)
         cam.calcHomographyWarped(frames, saveHomo = True, fileName="Cylindrical.npy")
        
      
   return {'status' : 200}

def get_frame():
   global cam
   print("Opening cam matrix")

   # open matrix outside of while loop to only do it once
   if numCams==3:
      print("Opening 3 cam matrix")
      Hl, Hr = cam.openHomographyFile("testFlaskHomography.npy")  
   elif numCams==6:
      #print("Opening 6 cam matrix")
         if Cylindrical and not noStitchJustStack:
             homoList = cam.openHomographyFile("Cylindrical.npy", "Cylindrical_Backup.npy")
         else:
            #Hl, Hr, Hl2, Hr2 = cam.openHomographyFile("testFlaskHomography6.npy")
            Hl, Hr = cam.openHomographyFile("testFlaskHomography0.npy", "testFlaskHomography.npy")
            print("Opening cam matrix 1")

            Hl2, Hr2 = cam.openHomographyFile("testFlaskHomography1.npy", "testFlaskHomography.npy")
            print("Opening cam matrix 2")
      
         
   while True:
      frames = cam.captureCameraImages()

      if noStitchJustStack:
         if Cylindrical:
            frames = cam.cylWarpFrames(frames)
         im = np.hstack(frames)
      else:
         if numCams == 1:
            im = frames[0]
         elif numCams == 3:
            im = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
         elif numCams == 6:
            if Cylindrical:

               im = cam.stitchWarped(frames, homoList)
               
            else:
               im = cam.homographyStitchTripleTwice(frames, Hl, Hr, Hl2, Hr2)

      imgencode=cv.imencode('.jpg',im)[1]
      stringData=imgencode.tobytes()
      yield (b'--frame\r\n'
         b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

def initialize():
   global cam
   print("Constructing camera system")
   if numCams == 1:
      print("Init 1 camera")
      cam = CameraSystem([0],compressCameraFeed=False,useLinuxCam=False) # laptop cam
   elif numCams == 3:
      print("Init 3 cameras")
      cam = CameraSystem([0,1,2], useLinuxCam=False)
   elif numCams == 4:
      print("Init 4 cameras")
      cam = CameraSystem(list(range(4)), useLinuxCam=False)
   elif numCams == 5:
      print("Init 5 cameras")
      cam = CameraSystem(list(range(5)), useLinuxCam=False)
   elif numCams == 6:
      print("Init 6 cameras")
      cam = CameraSystem(list(range(6)), useLinuxCam=True)

   print("Done init, calling recal")
   recalibrateCams()
   
       
@app.route('/vid',methods=['GET'])
def vid():
   global cam
   return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vid', methods=['POST'])
def recalibrateCamerasRoute():
   global cam
   recalibrateCams()
   return {'status' : 200}


@app.route('/panVid',methods=['GET'])
def getWebpage():
   print("Grabbing panellum page")
   return app.send_static_file('panellumStream.html')

@app.route('/',methods=['GET'])
def getCamWebpage():
   print("Grabbing cam page")
   return app.send_static_file('camHtmlRemote.html')

if __name__ == '__main__':
   try:
      initialize()
      app.run(host='192.168.55.1',port=5000, debug=False, threaded=True)
      #app.run(host='localhost',port=5000, debug=False, threaded=True)
   except Exception as e:
     print(e)
     print("Excpetion raised. Exiting...")
     GPIO.output(led_pin, GPIO.LOW)
     sys.exit(-1)
