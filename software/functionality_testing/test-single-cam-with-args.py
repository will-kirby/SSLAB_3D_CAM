import sys
import select
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Change Camera Recording Parameters.')
parser.add_argument('camera', metavar='X', type=int, help='Choose which camera to open')
#parser.add_argument('--duration', metavar='X.XX', type=float, )
args = parser.parse_args()

# print(sys.argv[1],sys.argv[2])
camNum = args.camera
camera = cv.VideoCapture(f'/dev/camera{camNum}')
#camera = cv.VideoCapture(camNum)
if not camera.isOpened():
    print(f"Cannot open camera {camNum}")
    exit()
capture_count = 0
while True:
    ret, frame = camera.read()
    # if frame is read correctly ret is True
    if not ret:
        print(f"Can't receive frame (stream end?). Exiting ...{camNum}")
        break

    cv.imshow('raw camera ' + str(camNum),frame)

 
    if cv.waitKey(1) == ord('c'):
        cv.imwrite(f'calibration/camera{camNum}_capture{capture_count}.png', frame)
        print(f'camera{camNum}_capture{capture_count}.png saved')
        capture_count += 1

    if cv.waitKey(2) == ord('q'):
        break
        
    if(select.select([sys.stdin,],[],[],0)[0] and sys.stdin.read(1) == 'q'): 
        break
camera.release()

cv.destroyAllWindows()
