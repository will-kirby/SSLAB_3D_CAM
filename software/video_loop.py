import numpy as np
import cv2 as cv
import imutils

import time
import matplotlib.pyplot as plt

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", type=str, required=True,
# 	help="path to input directory of images to stitch")
# ap.add_argument("-o", "--output", type=str, required=True,
# 	help="path to the output image")
# args = vars(ap.parse_args())

# # grab the paths to the input images and initialize our images list
# print("[INFO] loading images...")
# imagePaths = sorted(list(paths.list_images(args["images"])))

num_cameras = 2
cameras = None

jetson = False

if jetson:
    cameras = [cv.VideoCapture(f'dev/video{i}') for i in range(num_cameras)]
else:
    cameras = [cv.VideoCapture(i) for i in range(num_cameras)]

for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print(f"Cannot open camera {i}")
        exit()

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()

print("Press 'c' to capture image and quit")
print("Press 'q' to quit")

stitchTimes = []
stitcherStatuses = []

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

    stitcherStatuses.append(status)

    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        for i, frame in enumerate(frames):
            cv.imwrite(f'capture{i}.png', frame)
            print(f'capture{i}.png saved')
        break

# When everything done, release the capture
for camera in cameras:
    camera.release()

cv.destroyAllWindows()


# performance reporting
print(f'Percentage of dropped frames: {100 * np.count_nonzero(stitcherStatuses) / stitcherStatuses.size}%')
print(f'Average stitch time {np.mean(stitchTimes)}')

plt.hist(stitchTimes)
plt.show()