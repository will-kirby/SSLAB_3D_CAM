import sys
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Change Camera Recording Parameters.')
parser.add_argument('camera', metavar='X', type=int, help='Choose which camera to open')
#parser.add_argument('--duration', metavar='X.XX', type=float, )
args = parser.parse_args()

# print(sys.argv[1],sys.argv[2])
camNum = args.camera
camera = cv.VideoCapture("/dev/video"+str(camNum))
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
