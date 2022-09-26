from flask import Flask, render_template, Response
import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem

numCams = 1

print("Starting flask")
app = Flask(__name__)

print("Constructing camera system")
if numCams == 1:
    cam = CameraSystem([0],compressCameraFeed=False) # laptop cam
elif numCams == 2:
    cam = CameraSystem([1,2],compressCameraFeed=False) # two cams may need more work, i'm using Hl right now
elif numCams == 3:
    cam = CameraSystem([1,2,3])

print("Calculating homography for 0, 1, 2")
frames = cam.readFramesFromFiles(["capture0.png", "capture1.png","capture2.png"],"functionality_testing/")
Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])

# Save homo to file
print("Saving homo to file")
cam.saveHomographyToFile([Hl, Hr])

def get_frame():
    print("Opeing cam matrix")
    Hl, Hr = cam.openHomographyFile()

    while True:
        if numCams == 1:
            frames = cam.captureCameraImages()
            im = frames[0]
        elif numCams == 2:
            frames = cam.captureCameraImages()
            im = cam.cam.homographyStitch(frames[0], frames[1], Hl)
        elif numCams == 3:
            frames = cam.captureCameraImages()
            im = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)

        imgencode=cv.imencode('.jpg',im)[1]
        stringData=imgencode.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

@app.route('/vid',methods=['GET'])
def vid():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vid', methods=['POST'])
def toggleVidInput():
    print("Recalibrating input")
    frames = cam.captureCameraImages()
    if numCams == 3:
        Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        print("Saving homo to file")
        cam.saveHomographyToFile([Hl, Hr])
    elif numCams == 2:
        H, matchesMask = cam.calibrateMatrix()
        print("Saving homo to file")
        cam.saveHomographyToFile([H])
    elif numCams == 1:
        print("Num cams is 1, nothing to recalibrate")
    return {'status' : 200}

if __name__ == '__main__':
    app.run(host='localhost',port=5000, debug=True, threaded=True)