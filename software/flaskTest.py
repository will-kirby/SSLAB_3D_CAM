from flask import Flask, render_template, Response
import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem

numCams = 1
global cam
print("Starting flask")
app = Flask(__name__)

def get_frame():
    global cam
    print("Opening cam matrix")
    try:
       Hl, Hr = cam.openHomographyFile("testFlaskHomography.npy")
    
    except:
       Hl, Hr = cam.openHomographyFile("savedHomographyMatrix.npy") #use some back up homography
       
    while True:
        if numCams == 1:
            frames = cam.captureCameraImages()
            im = frames[0]
        elif numCams == 2:
            frames = cam.captureCameraImages()
            im = cam.homographyStitch(frames[0], frames[1], Hl)
        elif numCams == 3:
            frames = cam.captureCameraImages()
            im = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)

        imgencode=cv.imencode('.jpg',im)[1]
        stringData=imgencode.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

#@app.before_first_request
def initialize():
    global cam
    print("Constructing camera system")
    if numCams == 1:
       cam = CameraSystem([0],compressCameraFeed=False,useLinuxCam=False) # laptop cam
    elif numCams == 2:
       cam = CameraSystem([0,1],compressCameraFeed=False) # two cams may need more work, i'm using Hl right now
    elif numCams == 3:
       cam = CameraSystem([0,1,2])

    print("Calculating homography for 0, 1, 2")
    frames = cam.captureCameraImages()
    #frames = cam.readFramesFromFiles(["capture0.png", "capture1.png","capture2.png"],"../images/lab_5/")
    if numCams > 1:
      Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
      if (Hl is not None and Hr is not None):
         # Save homo to file
         print("Saving homo to file")
         cam.saveHomographyToFile([Hl, Hr],"testFlaskHomography.npy")
      
      else:
         print("Not enough matches detected to compute homography")
       
@app.route('/vid',methods=['GET'])
def vid():
     global cam
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vid', methods=['POST'])
def toggleVidInput():
    global cam
    print("Recalibrating input")
    frames = cam.captureCameraImages()
    if numCams == 3:
        Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        if (Hl is not None and Hr is not None):
           # Save homo to file
           print("Saving homo to file")
           cam.saveHomographyToFile([Hl, Hr],"testFlaskHomography.npy")
   
        else:
           print("Not enough matches detected to compute homography")
       
    elif numCams == 2:
        H, matchesMask = cam.calibrateMatrix()
        if (H is not None):
           # Save homo to file
           print("Saving homo to file")
           cam.saveHomographyToFile([H],"testFlaskHomography.npy")
   
        else:
           print("Not enough matches detected to compute homography")

    elif numCams == 1:
           print("Num cams is 1, nothing to recalibrate")
    return {'status' : 200}

if __name__ == '__main__':
    initialize()
   #  app.run(host='192.168.55.1',port=5000, debug=False, threaded=True)
    app.run(host='localhost',port=5000, debug=False, threaded=True)
