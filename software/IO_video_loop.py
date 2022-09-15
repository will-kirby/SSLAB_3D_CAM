import numpy as np
import cv2 as cv
import imutils
import time
import matplotlib.pyplot as plt
import subprocess
import sys
import argparse
import select

parser = argparse.ArgumentParser(description='Panorama Camera application')
parser.add_argument('-c', '--cameras', type=int, default=2,
help='Specify the amount of cameras')
parser.add_argument('-d', '--duration', type=int, default=10,
help='Specify the duration in seconds to run the test')
parser.add_argument('-i', '--image', action='store_true', default=False,
help='Flag to take one image')



# GPIO library
import Jetson.GPIO as GPIO
 
# Handles time
import time 
 
# Pin Definition
led_pin = 12
 
# Set up the GPIO channel
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH) 
 
print("Press CTRL+C when you want the LED to stop blinking") 

GPIO.output(led_pin, GPIO.HIGH) 
print("LED is ON")

num_cameras = 2
cameras = []# None

jetson = False

if jetson:
    cameras = [cv.VideoCapture(f'dev/video{i}') for i in range(num_cameras)]
else:
    for i in range(num_cameras):
        camera = cv.VideoCapture(i)
        camera.set(cv.CAP_PROP_FPS, 15)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        #camera.set(3, 320)# width
        #camera.set(3, 240)# height
        cameras.append(camera)

   # cameras = [cv.VideoCapture(i) for i in range(num_cameras)]

for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print(f"Cannot open camera {i}")
        exit()

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()

#video = cv.VideoWriter('test_vid.avi', cv.VideoWriter_fourcc(*'MJPG'),30, (320,240))
video = cv.VideoWriter('test_stitched_vid.avi', cv.VideoWriter_fourcc(*'MJPG'),3, (640,480) )

print("Press 'c' to capture image and quit")
print("Press 'q' to quit")

stitchTimes = []
stitcherStatuses = []

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
        #if(i == 1):
          #  video.write(frame)
    if not ret:
        continue

    # Display the resulting frame
    #cv.imshow('raw camera', np.concatenate(frames, axis=1))
    
    # Our operations on the frame come here
    timeStart = time.perf_counter()
    (status, stitched) = stitcher.stitch(frames)
    
    
    if status==0:
        #cv.imshow('stitched', stitched)
        stitchTimes.append(time.perf_counter() - timeStart)
        stitched_resize = cv.resize(stitched, (640,480))
        video.write(stitched_resize)
        print("stitched!")
        

    stitcherStatuses.append(status)
    if(select.select([sys.stdin,],[],[],0)[0] and sys.stdin.read(1) == 'q'): #poll stdin for a character (piped in from open_gui.py)
        break
    if cv.waitKey(1) == ord('q'):
        break
    if cv.waitKey(1) == ord('c'):
        for i, frame in enumerate(frames):
            cv.imwrite(f'capture{i}.png', frame)
            cv.imwrite(f'stitched.png', stitched)
            print(f'capture{i}.png saved')
        break

# When everything done, release the capture
for camera in cameras:
    camera.release()

video.release();
cv.destroyAllWindows()

# Deprecated for log parsing
# # performance reporting
# print(f'Percentage of dropped frames: {100 * np.count_nonzero(stitcherStatuses) / len(stitcherStatuses)}%')
# mean = np.mean(stitchTimes)

# plt.hist(stitchTimes, density=True)
# plt.axvline(mean, color='k', linestyle='dashed', linewidth=1,label=(f'mean={mean}'))
# plt.title(f'Stitch Times: Percentage of dropped frames: {100*np.count_nonzero(stitcherStatuses) / len(stitcherStatuses)}%')
# plt.ylabel('Density')
# plt.xlabel('Time (seconds)')
# plt.legend()
# plt.savefig('../figures/Stitch_Time_Hist_case.png')

# plt.show()

#Begin File Transfer
args = sys.argv
args[0] = "./common/copy.sh" # path to shell script
print("Trasnfering file " + args[2])
subprocess.check_call(args)
