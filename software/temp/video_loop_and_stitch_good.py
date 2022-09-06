# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2 as cv
import imutils


# from stitchTest import StitchImages

# cameras = []

camera_0 = cv.VideoCapture(0)
camera_1 = cv.VideoCapture(1)
camera_2 = cv.VideoCapture(2)
camera_3 = cv.VideoCapture(3)


camera_0.set(cv.CAP_PROP_FPS, 30)
camera_0.set(cv.CAP_PROP_FRAME_WIDTH, 320)
camera_0.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
camera_0.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

camera_1.set(cv.CAP_PROP_FPS, 30)
camera_1.set(cv.CAP_PROP_FRAME_WIDTH, 320)
camera_1.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
camera_1.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

camera_2.set(cv.CAP_PROP_FPS, 30)
camera_2.set(cv.CAP_PROP_FRAME_WIDTH, 320)
camera_2.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
camera_2.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

camera_3.set(cv.CAP_PROP_FPS, 30)
camera_3.set(cv.CAP_PROP_FRAME_WIDTH, 320)
camera_3.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
camera_3.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

if not camera_0.isOpened():
    print("Cannot open camera")
    exit()

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()


index = 0
while True:
    # Capture frame-by-frame
    ret0, frame0 = camera_0.read()
    frame0_disp = cv.resize(frame0, (240, 320)) # resize to fit in one window
    ret1, frame1 = camera_1.read()
    frame1_disp = cv.resize(frame1, (240, 320))
    ret2, frame2 = camera_2.read()
    frame2_disp = cv.resize(frame2, (240, 320))
    ret3, frame3 = camera_3.read()
    frame3_disp = cv.resize(frame3, (240, 320))

    # print("Cam 1",frame0.shape,"Cam 2",frame1.shape)

    
    # if frame is read correctly ret is True
    if not (ret0 and ret1 and ret2):
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('raw camera', np.concatenate((frame0_disp, frame1_disp, frame2_disp, frame3_disp), axis=1))
    
    (status, stitched) = stitcher.stitch((frame0, frame1, frame2, frame3))
    
    if status==0:
        cv.imshow('stitched', stitched)


    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
camera_0.release()
camera_1.release()
camera_2.release()
camera_3.release()


cv.destroyAllWindows()
