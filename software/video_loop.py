# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2 as cv
import imutils


# from stitchTest import StitchImages

# cameras = []

camera_0 = cv.VideoCapture(0)
camera_1 = cv.VideoCapture(1)
camera_2 = cv.VideoCapture(2)


if not camera_0.isOpened():
    print("Cannot open camera")
    exit()

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()


index = 0
while True:
    # Capture frame-by-frame
    ret, frame0 = camera_0.read()
    frame0_disp = cv.resize(frame0, (240, 320))
    ret, frame1 = camera_1.read()
    frame1_disp = cv.resize(frame1, (240, 320))
    ret, frame2 = camera_2.read()
    frame2_disp = cv.resize(frame2, (240, 320))

    # print("Cam 1",frame0.shape,"Cam 2",frame1.shape)

    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('raw camera', np.concatenate((frame0_disp, frame1_disp, frame2_disp), axis=1))
    
    (status, stitched) = stitcher.stitch((frame0, frame1, frame2))
    
    if status==0:
        cv.imshow('stitched', stitched)


    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        cv.imwrite('capture' + str(index) + '.png', frame)
        print('Image saved')
        index = index + 1
# When everything done, release the capture
camera_0.release()
camera_1.release()

cv.destroyAllWindows()