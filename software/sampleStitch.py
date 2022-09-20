from CameraSystemClass import CameraSystem
import cv2 as cv
import numpy as np



print("Constructing camera system")
cam = CameraSystem([2,1],compressCameraFeed=False)
# cam = CameraSystem([2,1,3])

print("Calculating homography for 0 and 1")
H, matchesMask = cam.calibrateMatrix()

# Save homo
cam.saveHomographyToFile([H])

# Open saved matrix
H = cam.openHomographyFile()


while(1):
    # capture images
    frames = cam.captureCameraImages()

    # Display the resulting frame
    cv.imshow('raw', np.concatenate(frames, axis=1))

    # Stitch
    stitched = cam.homographyStitch(frames[0], frames[1], H[0])
    cv.imshow('stitched', stitched)

    if cv.waitKey(1) == ord('r'):
        print("Recalibrating")
        H, matchesMask = cam.calibrateMatrix()
        print("Done recalibrating")

    if cv.waitKey(1) == ord('q'):
        break
