# doesn't seem to work, I was getting "Cannot open camera every time...."

import sys
import cv2 as cv

# print(sys.argv[1],sys.argv[2])
camNum = sys.argv[1]
camera = cv.VideoCapture(camNum)
if not camera.isOpened():
    print(f"Cannot open camera {camNum}")
    exit()

while True:
    ret, frame = camera.read()
    # if frame is read correctly ret is True
    if not ret:
        print(f"Can't receive frame (stream end?). Exiting ...{camNum}")
        break

    cv.imshow('raw camera ' + str(camNum),frame)

    if cv.waitKey(1) == ord('q'):
        break

camera.release()

cv.destroyAllWindows()