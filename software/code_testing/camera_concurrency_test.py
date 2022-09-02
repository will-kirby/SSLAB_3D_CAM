import cv2 as cv
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Please specify a camera to open")
    exit()

number = sys.argv[1]
print(f'Opening camera {number}')

camera = cv.VideoCapture(number)
camera.set(3, 200)

if not camera.isOpened():
    print(f"Cannot open camera {number}")
else:
    print("Reading. Press 'q' to exit")

while True:
    frames = []

    # Capture frame-by-frame
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

camera.release()

cv.destroyAllWindows()