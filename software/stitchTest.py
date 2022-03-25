# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2 as cv
import imutils

from statistics import mean
import time

# cameras = []
# for i in range(0,3):
#     cameras.append(cv.VideoCapture('/dev/video' + str(i)))
#     if not cameras[i].isOpened():
#         print("Cannot open camera " + str(i))
#         exit()

# index = 0
# for camera in cameras:
#     ret, frame =  camera.read()
#     cv.imwrite('capture' + str(index) + '.png', frame)
#     camera.release()

readTimes = []
imagePaths = ['capture0.png','capture1.png','capture2.png']

images = []
for imagePath in imagePaths:
	t1 = time.perf_counter_ns()
	image = cv.imread(imagePath)
	readTimes.append(time.perf_counter_ns() - t1)
	images.append(image)

print(f'Average read time: {mean(readTimes) / 1000000000} seconds')

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
t1 = time.perf_counter_ns()
(status, stitched) = stitcher.stitch(images)
print(f'Time to stitch images: {(time.perf_counter_ns() - t1) / 1000000000} seconds')

# if the status is '0', then OpenCV successfully performed image
if status == 0:
	# write the output stitched image to disk
	cv.imwrite('stitchedImage.png', stitched)