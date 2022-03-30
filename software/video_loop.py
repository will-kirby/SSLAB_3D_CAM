import numpy as np
import cv2 as cv

num_cameras = 3
cameras = [cv.VideoCapture(i) for i in range(num_cameras)]

for i, camera in enumerate(cameras):
    if not camera.isOpened():
        print(f"Cannot open camera {i}")
        exit()

print("Press 'q' to quit")
print("Press 'c' to save images and quit")

while True:
    # Capture frame-by-frame
    # frames = [camera.read()[1] for camera in cameras]
    frames = []
    for camera in cameras:
        ret, frame = camera.read()
        frames.append(frame)
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

    # Our operations on the frame come here (middleware)

    # stitcher = cv.Stitcher_create()  # if imutils.is_cv3() else cv.Stitcher_create()
    # (status, stitched) = stitcher.stitch([frame, frame2])

    frames = np.concatenate(frames, axis=1) # change to stitcher

    # Display the resulting frame
    cv.imshow('frame', frames)
    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        for i, frame in enumerate(frames):
            cv.imwrite(f'capture{i}.png', frame)
            print('Image saved')
        break

# When everything done, release the capture
for camera in cameras:
    camera.release()

cv.destroyAllWindows()