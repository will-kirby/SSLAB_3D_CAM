import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time
import select
import sys

"""
    Currently uses pre-taken images for timing analysis.
"""

useStaticImages = True
imagePaths = ["../images/" + p for p in [
    "lab_5/capture", # pngs
    "lab_6/capture",
    "lab_7/capture",
    "labPaper/0/image", # jpgs
    "labPaper/1/image",
    "labPaper/2/image",
]]
pngSet = {0,1,2} # if change the above, make sure it is in here
cameraSet = [[0,1,2],[3,4,5]] # cameras 0-2 and 3-5 at a time
imagePathIndex = 0 # initial indices
cameraSetIndex = 0

initialFocal = 200
focusList = [initialFocal for i in range(6)]

cam = CameraSystem([],compressCameraFeed=False)

message = """
q,w,e to increase focal length
a,s,d to decrease focal length
t to toggle camera set (cameras 0-2,3-5)
n to change image set
p to quit
l to list focal lengths
h to print this
"""
print(message)

def getFrames():
    return cam.readFramesFromFiles([str(i) + (".png" if imagePathIndex in pngSet else ".jpg") 
                                    for i in cameraSet[cameraSetIndex]],
                                    imagePaths[imagePathIndex])
def makePano(images):
    cylMatrices = cam.getCylCoords(images, focalLengths = [focusList[i] for i in cameraSet[cameraSetIndex]])
    warpedFrames = cam.applyCylWarp(images, cylMatrices)
    warpedFrames[1] = cam.borderImg(warpedFrames[1])
    HlL, HrL = cam.calcHomographyThree(warpedFrames[0], warpedFrames[1], warpedFrames[2])
    if HlL is None or HrL is None:
        return None
    return cam.stitchThree(warpedFrames[0], warpedFrames[1], warpedFrames[2], HlL, HrL)

def showImg(img):
    cv.imshow("pano",img)

def fullUpdate():
    cameraImages = getFrames()
    pano = makePano(cameraImages)
    if pano is not None:
        showImg(pano)

def focalChangeUpdate(index, amount):
    focusList[cameraSet[cameraSetIndex][index]] += amount
    print(f"Camera Index {cameraSet[cameraSetIndex][index]} {'increased' if amount == 1 else 'decreased'} to {focusList[cameraSet[cameraSetIndex][index]]}")

    fullUpdate()

fullUpdate()
while True:

    keyPressed = cv.waitKey(1)
    if keyPressed == ord('p'):
        break
    elif keyPressed == ord('n'):
        imagePathIndex = (imagePathIndex + 1) % len(imagePaths)
        print("image path changed",imagePathIndex, imagePaths[imagePathIndex])
        fullUpdate()
    elif keyPressed == ord('t'):
        cameraSetIndex = (cameraSetIndex + 1) % len(cameraSet)
        print("Camera set changed",cameraSetIndex,cameraSet[cameraSetIndex])
        fullUpdate()

    # focal length changes
    elif keyPressed == ord('q'):
        focalChangeUpdate(0, 1)
    elif keyPressed == ord('w'):
        focalChangeUpdate(1, 1)
    elif keyPressed == ord('e'):
        focalChangeUpdate(2, 1)
    elif keyPressed == ord('a'):
        focalChangeUpdate(0, -1)
    elif keyPressed == ord('s'):
        focalChangeUpdate(1, -1)
    elif keyPressed == ord('d'):
        focalChangeUpdate(2, -1)

    elif keyPressed == ord('l'):
        print(focusList)
    elif keyPressed == ord('h'):
        print(message)

cv.destroyAllWindows()