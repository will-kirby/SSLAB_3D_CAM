import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time

print("Constructing start")
cam = CameraSystem(range(numCams), useLinuxCam=True)

print("taking images")
frames = cam.captureCameraImages()

randNum = int(time.time()/10000)

for i, frame in enumerate(frames):
    cv.imwrite(f'../images/labPaper/{randNum}/image{i}.jpg', frame)

print(f"Done, wrote to ../images/labPaper/{randNum}/image[x]")