import numpy as np
import cv2 as cv
import imutils

import time
import matplotlib.pyplot as plt

num_cameras = 2
cameras = []# None

for i in range(num_cameras):
  print(i)
  camera = cv.VideoCapture('/dev/video'+str(i))
  camera.set(3, 320)# width
  camera.set(3, 240)# height
  cameras.append(camera)


for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()

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
            print("Can't receive frame (stream end?). Exiting ...")
            break
    if not ret:
        continue

    # Display the resulting frame
    cv.imshow('raw camera', np.concatenate(frames, axis=1))
    
    # Our operations on the frame come here
    timeStart = time.perf_counter()
    (status, stitched) = stitcher.stitch(frames)
    
    
    if status==0:
        cv.imshow('stitched', stitched)
        stitchTimes.append(time.perf_counter() - timeStart)

    stitcherStatuses.append(status)

    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        for i, frame in enumerate(frames):
            cv.imwrite('capture'+str(i)+'.png', frame)
        break

# When everything done, release the capture
for camera in cameras:
    camera.release()

cv.destroyAllWindows()


# performance reporting
print('Percentage of dropped frames: '+str(100 * np.count_nonzero(stitcherStatuses) / len(stitcherStatuses))+'%')
mean = np.mean(stitchTimes)

plt.hist(stitchTimes, density=True)
plt.axvline(mean, color='k', linestyle='dashed', linewidth=1,label=('mean='+str(mean)))
plt.title('Stitch Times: Percentage of dropped frames: '+str(100*np.count_nonzero(stitcherStatuses) / len(stitcherStatuses))+'%')
plt.ylabel('Density')
plt.xlabel('Time (seconds)')
plt.legend()
plt.savefig('../figures/Stitch_Time_Hist_case.png')

plt.show()
