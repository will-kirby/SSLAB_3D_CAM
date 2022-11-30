from flask import Flask, render_template, Response
import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem
import Jetson.GPIO as GPIO
import signal

numCams = 6
led_pin=12
noStitchJustStack = False
Cylindrical = True
cylWarpInitial = 146
cylWarpIncrement = 6
focalLengths = [122, 138, 133, 128, 119, 116] #[108, 118, 130, 146, 132, 123]
cutThenAppendFirst = True
origin2Stitch = True # if false uses left to right

# change this to True on actual thing?
recalibrateOnInit = False
global cam
global cylMatrices

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
   global cylMatrices
   print("Recalibrating cams")
   frames = cam.captureCameraImages()
   if Cylindrical:
      if origin2Stitch:
         frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend = True, borderOnFirstAndFourth=True)
         cam.calcHomographyWarped2Origin(frames)
      else:
         #frames = cam.cylWarpFrames(frames, focalLength=focalLength,incrementAmount=cylWarpIncrement, cutFirstThenAppend=cutFirstThenAppend)
         frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend=cutFirstThenAppend)
         #frames = cam.cylWarpFrames(frames, cylWarpInitial, cylWarpIncrement, cutFirstThenAppend=cutThenAppendFirst)
         cam.calcHomographyWarped(frames, saveHomo = True, fileName=f"Cylindrical{numCams}.npy")
   else:

      if numCams == 1:
         print("Num cams is 1, nothing to recalibrate")

      elif numCams == 3:
         Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2], save=True, filename="testFlaskHomography.npy")

      elif numCams == 6:
         #Hl, Hr, Hl2, Hr2 = cam.calibrateMatrixTripleTwice(frames, save=True, filename="testFlaskHomography6.npy")
         Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2], save=True, filename="testFlaskHomography0.npy")
         Hl2, Hr2 = cam.calibrateMatrixTriple(frames[3], frames[4], frames[5], save=True, filename="testFlaskHomography1.npy")
        
      
   return {'status' : 200}

def get_frame():
   global cam
   global cylMatrices    
   print("Opening cam matrix")
   
   if not noStitchJustStack:
   # open matrix outside of while loop to only do it once
      if Cylindrical:
            print(f"Opening cylindrical homography matrices for {numCams} cameras")
            if origin2Stitch:
                homoList = cam.openHomographyFile("homography2origin.npy", "homography2origin_Backup.npy")
            else:
                homoList = cam.openHomographyFile(f"Cylindrical{numCams}.npy", f"Cylindrical{numCams}_Backup.npy")
      else:
         if numCams==3:
            print("Opening 3 cam matrix")
            Hl, Hr = cam.openHomographyFile("testFlaskHomography.npy")  
         elif numCams==6:
            print("Opening 6 cam matrix")
    
            #Hl, Hr, Hl2, Hr2 = cam.openHomographyFile("testFlaskHomography6.npy")
            Hl, Hr = cam.openHomographyFile("testFlaskHomography0.npy", "testFlaskHomography.npy")
            print("Opening cam matrix 1")

            Hl2, Hr2 = cam.openHomographyFile("testFlaskHomography1.npy", "testFlaskHomography.npy")
            print("Opening cam matrix 2")
      
         
   while True:
      frames = cam.captureCameraImages()
      if Cylindrical:
            if origin2Stitch:
                frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend = True, borderOnFirstAndFourth=True)
            else:
                frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend=cutFirstThenAppend)
        
      # hstack frames, no stitching
      if noStitchJustStack:
         im = np.hstack(frames)
  
      # stitch with cylindrical warp
      elif Cylindrical:
            if origin2Stitch:
                homoList = cam.openHomographyFile("homography2origin.npy", "homography2origin_Backup.npy")
                im = cam.stitchWarped2Origin(frames, homoList)
            else:
                homoList = cam.openHomographyFile(f"Cylindrical{numCams}.npy", f"Cylindrical{numCams}_Backup.npy")
                im = cam.stitchWarped(frames, homoList)
            #im = cv.resize(im, dsize=(640,480), interpolation = cv.INTER_LINEAR)
      # stitch, no warping
      else:
         if numCams == 1:
            im = frames[0]
         elif numCams == 3:
            im = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
         elif numCams == 6:
            im = cam.homographyStitchTripleTwice(frames, Hl, Hr, Hl2, Hr2)

      imgencode=cv.imencode('.jpg',im)[1]
      stringData=imgencode.tobytes()
      yield (b'--frame\r\n'
         b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

def initialize():
   global cam
   global cylMatrices
   print("Constructing camera system")
   print(f"Init {numCams} cameras")
   cam = CameraSystem(range(numCams), useLinuxCam=True)
   frames = cam.captureCameraImages()
   cylMatrices = cam.getCylCoords(frames, cylWarpInitial, cylWarpIncrement, focalLengths=focalLengths)

   print("Done init, calling recal")
   recalibrateCams()
   
       
@app.route('/vid',methods=['GET'])
def vid():
   return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vid', methods=['POST'])
def recalibrateCamerasRoute():
   recalibrateCams()
   return {'status' : 200}

@app.route('/reset', methods=['POST'])
def recalibrateCamerasRoute():
   global cam
   try:
      homography = cam.openHomographyFile("homography2origin_Backup.npy")
      cam.saveHomographyToFile("homography2origin.npy")
      return {'status' : 200}
   except:
      return {'status' : 400}

@app.route('/blur', methods=['POST'])
def recalibrateCamerasRoute():
   global cam
   try:
      cam.blend = not cam.blend
      return {'status' : 200}
   except:
      return {'status' : 400}

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
     print("Exception raised. Exiting...")
     GPIO.output(led_pin, GPIO.LOW)
     sys.exit(-1)
