# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2 as cv

# from stitchTest import StitchImages

# cameras = []

camera_0 = cv.VideoCapture(0)
if not camera_0.isOpened():
    print("Cannot open camera")
    exit()

index = 0
while True:
    # Capture frame-by-frame
    ret, frame = camera_0.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        cv.imwrite('capture' + str(index) + '.png', frame)
        print('Image saved')
        index = index + 1
# When everything done, release the capture
camera_0.release()
cv.destroyAllWindows()