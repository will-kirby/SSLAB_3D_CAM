import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem

print("Constructing start")
cam = CameraSystem(range(numCams), useLinuxCam=True)

print("taking images")
frames = cam.captureCameraImages()

for i, frame in enumerate(frames):
    cv.imwrite(f'imagesForPano/image{i}.jpg', frame)

print("Done, wrote to imagesForPano/image[x]")