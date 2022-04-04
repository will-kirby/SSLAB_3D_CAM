import numpy as np
import cv2 as cv
import imutils

import time

num_cameras = 3
cameras = [cv.VideoCapture(i) for i in range(num_cameras)]

for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print(f"Cannot open camera {i}")
        exit()

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()

print("Press 'c' to capture image and quit")
print("Press 'q' to quit")

stitchTimes = []
droppedFrames = 0

while True:
    frames = []

    # Capture frame-by-frame
    for camera in cameras:
        ret, frame = camera.read()
        frames.append(cv.resize(frame, (240, 320)))
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break


    # Display the resulting frame
    cv.imshow('raw camera', np.concatenate(frames, axis=1))
    
    # Our operations on the frame come here
    timeStart = time.perf_counter()
    (status, stitched) = stitcher.stitch(frames)
    stitchTimes.append(time.perf_counter() - timeStart)
    
    if status==0:
        cv.imshow('stitched', stitched)
    else:
        droppedFrames += 1


    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        for i, frame in enumerate(frames):
            cv.imwrite(f'capture{i}.png', frame)
            print('Image saved')
        break

print(f'Average stitch time {np.mean(stitchTimes)}')

# When everything done, release the capture
for camera in cameras:
    camera.release()

cv.destroyAllWindows()