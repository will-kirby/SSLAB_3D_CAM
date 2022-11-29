import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time
import os


numCams = 6
baseDir='../images/labPaper/'
print("Constructing start")
cam = CameraSystem(range(numCams), useLinuxCam=True)

print("taking images")
frames = cam.captureCameraImages()

Num = 0
os.chdir(baseDir)

while(os.path.exists(f"{Num}")):
    Num += 1

dirname = f'{Num}/'
os.mkdir(dirname)

for i, frame in enumerate(frames):
    cv.imwrite(dirname + f'image{i}.jpg', frame)

print(f"Done, wrote to ../images/labPaper/{Num}/image[x]")
