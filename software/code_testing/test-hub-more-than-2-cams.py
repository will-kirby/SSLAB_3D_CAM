# on windows machine, I was able to get 4 camears to work: 2 on hub,
#   1 webcam, and 1 plugged in sperately from the hub
# When i Tried 3 on the hub, I was given an error
# On windows, can go to device manager -> cameras to see attached cameras
# It apperas that 0 is the first camera, 1 is the second, etc. but not 100% on this

import cv2 as cv
import numpy as np

cameras = []
num_cameras = 4 # on my laptop, this worked, used webcam, 2 hub, and 1 sep plugged in
startIndex = 0 # if on laptop, avoid the webcam (0 for jetson)
compress = 0 # change whether to use the compressed camera feed

print("Staring program...")
print(f"Number of cameras={num_cameras}, Starting Index={startIndex}, Compress camera feed={compress}")
for i in range(num_cameras):
    camera = cv.VideoCapture(i+startIndex)
    if compress:
        camera.set(cv.CAP_PROP_FPS, 30)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    else:
        camera.set(3, 200)# width

    cameras.append(camera)

    #cameras = [cv.VideoCapture(i) for i in range(num_cameras)]
print("Cameras constructed")

for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print(f"Cannot open camera {i}")
        exit()
print(f"{num_cameras} successfully opened")


# print("Press 'c' to capture image and quit")
print("Press 'q' to quit")
print("Press 's' to push images together")

shiftAmount = 0
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

    # calculate pushed (shape[0] is height, shape[1] is width)
    pushedImages = np.zeros((max([frame.shape[0] for frame in frames]), sum([frame.shape[1] for frame in frames]), 3), dtype="uint8" )
    
    currentWidth = 0
    for i, frame in enumerate(frames):
        pushedImages[0:frame.shape[0], (currentWidth-shiftAmount*i):(currentWidth + frame.shape[1]-shiftAmount*i)] = frame
        currentWidth += (frame.shape[1] - shiftAmount)
    cv.imshow('pushed images', pushedImages)


    if cv.waitKey(1) == ord('q'):
        break

    # if cv.waitKey(1) == ord('c'):
    #     for i, frame in enumerate(frames):
    #         cv.imwrite(f'capture{i}.png', frame)
    #         print(f'capture{i}.png saved')
    #     break

    if cv.waitKey(1) == ord('s'):
        shiftAmount += 5
        print(f"Incrementing shift amount, current amount: {shiftAmount}")

# When everything done, release the capture
for camera in cameras:
    camera.release()

cv.destroyAllWindows()
