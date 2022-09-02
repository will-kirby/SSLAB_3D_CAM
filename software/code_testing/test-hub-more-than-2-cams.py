# on windows machine, I was able to get 4 camears to work: 2 on hub,
#   1 webcam, and 1 plugged in sperately from the hub
# When i Tried 3 on the hub, I was given an error
# On windows, can go to device manager -> cameras to see attached cameras
# It apperas that 0 is the first camera, 1 is the second, etc. but not 100% on this

import cv2 as cv
import numpy as np

cameras = []
num_cameras = 4 # on my laptop, this worked, used webcam, 2 hub, and 1 sep plugged in
for i in range(num_cameras):
    camera = cv.VideoCapture(i)
    camera.set(3, 200)# width
    cameras.append(camera)

    #cameras = [cv.VideoCapture(i) for i in range(num_cameras)]

for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print(f"Cannot open camera {i}")
        exit()

print("Press 'c' to capture image and quit")
print("Press 'q' to quit")

while True:
    frames = []

    # Capture frame-by-frame
    for i, camera in enumerate(cameras):
        ret, frame = camera.read()
        frames.append(frame)
        # if frame is read correctly ret is True
        if not ret:
            print(f"Can't receive frame (stream end?). Exiting ...{i}")
            break

    # Display the resulting frame
    cv.imshow('raw camera', np.concatenate(frames, axis=1))

    if cv.waitKey(1) == ord('q'):
        break

    # if cv.waitKey(1) == ord('c'):
    #     for i, frame in enumerate(frames):
    #         cv.imwrite(f'capture{i}.png', frame)
    #         print(f'capture{i}.png saved')
    #     break


# When everything done, release the capture
for camera in cameras:
    camera.release()

cv.destroyAllWindows()