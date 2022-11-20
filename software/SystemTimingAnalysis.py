import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time

"""
    Currently uses pre-taken images for timing analysis.
"""

useStaticImages = True

focalLength = 195
# imageIndex = [2,3,4,5,0,1]
imageIndex = range(6)
labNum=5

print("Starting timing anal prog")
# construct cam system and grab initial frames
if useStaticImages:
    cam = CameraSystem([],compressCameraFeed=False)
    frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],f"../images/lab_{labNum}/capture")
else:
    cam = CameraSystem(range(numCams), useLinuxCam=True)
    frames = cam.captureCameraImages()

# warp frames
frames = cam.cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=True, borderOnFirstAndFourth=True)

# calculate initial homography
print("Calculating homography")
pt = time.time()
homoList = cam.calcHomographyWarped2Origin(frames)
etH = time.time() - pt
print(" - Elapsed time: ",etH)

while True:
    times = []
    initTime = time.time()

    # grab frames
    pt = time.time()

    if useStaticImages:
        frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],f"../images/lab_{labNum}/capture")
    else:
        frames = cam.captureCameraImages()

    times.append(time.time()-pt)
    

    # warp frames
    pt = time.time()

    frames = cam.cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=True, borderOnFirstAndFourth=True)

    times.append(time.time()-pt)


    # stitch frames
    pt = time.time()

    pano = cam.stitchWarped2Origin(frames, homoList)

    times.append(time.time()-pt)

    for i, t in enumerate(times):
        times[i] = round(t,4)

    per = time.time() - initTime
    frameRate = round(1/per,2)

    # Show pano, frame rate, individual times
    font = cv.FONT_HERSHEY_SIMPLEX
    size = .6
    x,y = 10, 500
    inc = 20
    # cv.putText(pano, f"Grab: {times[0]}s  Warp: {times[1]}s  Stitch: {times[2]}s", (100,80), font, 0.75,(0,0,255),2)

    cv.putText(pano, f"Grab frames: {times[0]} s  ", (x,y), font, size,(0,0,255),2)
    cv.putText(pano, f"Warp frames: {times[1]} s  ", (x,y+inc), font, size,(0,0,255),2)
    cv.putText(pano, f"Stitch frames: {times[2]} s  ", (x,y+inc*2), font, size,(0,0,255),2)
    cv.putText(pano, f"Frame Rate: {frameRate} FPS", (x,y+inc*3), font, size,(0,0,255),2)
    cv.putText(pano, f"Homo Calc: {round(etH,4)} s", (x,y+inc*4), font, size,(0,0,255),2)


    cv.imshow("Pano",pano)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()



    

