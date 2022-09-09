import cv2 as cv
import numpy as np
import sys
from time import sleep

if len(sys.argv) < 2:
    print("Please specify a camera to open")
    exit()

number = sys.argv[1]
print(f'Opening camera {number}')

camera = cv.VideoCapture(f'/dev/video{number}')
camera.set(cv.cv.CAP_PROP_FPS, 15)
camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

if not camera.isOpened():
    print(f"Cannot open camera {number}")
else:
    print(f"Reading camera {number}. Press 'q' to exit")

while True:

    # Capture frame-by-frame
    ret, frame = camera.read()
    # if frame is read correctly ret is True
    if not ret:
        print(f"Can't receive frame (stream end?). Exiting...")
        break

    # Display the resulting frame
    cv.imshow(f'raw camera {number}', frame)
    #print("captured frame")
	
    #sleep(1)

    if cv.waitKey(1) == ord('q'):
        break

camera.release()

cv.destroyAllWindows()
