import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time
import os


numCams = 6
print("Constructing start")
cam = CameraSystem(range(numCams), useLinuxCam=True)

print("taking images")
frames = cam.captureCameraImages()

randNum = 2#int(time.time()/10000)

os.chdir('../images')
os.makedirs(f'labPaper/{randNum}/', exist_ok=True)
dirname = f'labPaper/{randNum}/'
for i, frame in enumerate(frames):
    cv.imwrite(dirname + f'image{i}.jpg', frame)

print(f"Done, wrote to ../images/labPaper/{randNum}/image[x]")
