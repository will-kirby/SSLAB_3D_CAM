from CameraSystemClass import CameraSystem
import cv2 as cv
import numpy as np



print("Constructing camera system")
cam = CameraSystem([2,1],compressCameraFeed=False)
# cam = CameraSystem([0,1,2])


print("Calculating homography for 0 and 1")
H, matchesMask = cam.calibrateMatrix()
# Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])

# Save homo
cam.saveHomographyToFile([H])
# cam.saveHomographyToFile([Hl, Hr])

# Open saved matrix
H = cam.openHomographyFile()
# Hl, Hr = cam.openHomographyFile()


while(1):
    # capture images
    frames = cam.captureCameraImages()

    # Display the resulting frame
    cv.imshow('raw', np.concatenate(frames, axis=1))

    # Stitch
    stitched = cam.homographyStitch(frames[0], frames[1], H[0])
    # stitched = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
    cv.imshow('stitched', stitched)

    if cv.waitKey(1) == ord('r'):
        print("Recalibrating")
        H, matchesMask = cam.calibrateMatrix()
        # Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        print("Done recalibrating")

    if cv.waitKey(1) == ord('q'):
        break
