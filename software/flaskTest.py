from flask import Flask, render_template, Response
import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem

numCams = 6
noStitchJustStack = False
global cam
print("Starting flask")
app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static')

def recalibrateCams():
   global cam
   print("Recalibrating cams")
   frames = cam.captureCameraImages()

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
   print("Opening cam matrix")

   # open matrix outside of while loop to only do it once
   if numCams==3:
      print("Opening 3 cam matrix")
      Hl, Hr = cam.openHomographyFile("testFlaskHomography.npy")  
   elif numCams==6:
      #print("Opening 6 cam matrix")
         #Hl, Hr, Hl2, Hr2 = cam.openHomographyFile("testFlaskHomography6.npy")
         Hl, Hr = cam.openHomographyFile("testFlaskHomography0.npy", "testFlaskHomography.npy")
         print("Opening cam matrix 1")

         Hl2, Hr2 = cam.openHomographyFile("testFlaskHomography1.npy", "testFlaskHomography.npy")
         print("Opening cam matrix 2")
      
         
   while True:
      frames = cam.captureCameraImages()

      if noStitchJustStack:
         im = np.hstack(frames)
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
   print("Constructing camera system")
   if numCams == 1:
      print("Init 1 camera")
      cam = CameraSystem([0],compressCameraFeed=False,useLinuxCam=False) # laptop cam
   elif numCams == 3:
      print("Init 3 cameras")
      cam = CameraSystem([0,1,2])
   elif numCams == 6:
      print("Init 6 cameras")
      cam = CameraSystem(list(range(6)))

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
   return app.send_static_file('panellum.html')

@app.route('/',methods=['GET'])
def getCamWebpage():
   print("Grabbing cam page")
   return app.send_static_file('cams.html')

if __name__ == '__main__':
   initialize()
   app.run(host='192.168.55.1',port=5000, debug=False, threaded=True)
   #app.run(host='localhost',port=5000, debug=False, threaded=True)
