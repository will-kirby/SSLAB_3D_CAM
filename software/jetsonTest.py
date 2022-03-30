# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2 as cv
import imutils


cameras = []
for i in range(0,3):
    cameras.append(cv.VideoCapture('/dev/video' + str(i)))
    if not cameras[i].isOpened():
        print("Cannot open camera " + str(i))
        exit()

index = 0
for camera in cameras:
    ret, frame =  camera.read()
    cv.imwrite('capture' + str(index) + '.png', frame)
    index = index + 1
    camera.release()

imagePaths = ['capture0.png','capture1.png','capture2.png']

images = []
for imagePath in imagePaths:
	image = cv.imread(imagePath)
	images.append(image)

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
if status == 0:
	# write the output stitched image to disk
	cv.imwrite('stitchedImage.png', stitched)
    