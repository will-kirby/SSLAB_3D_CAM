import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time

"""
    Currently uses pre-taken images for timing analysis.
"""

useStaticImages = False
origin2Stitch = False # if false uses left to right
numCams=4
focalLength = 195
cylWarpIncrement = 4
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

    #frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],f"../images/lab_{labNum}/capture")

# warp frames
if origin2Stitch:
    frames = cam.cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=True, borderOnFirstAndFourth=True)
else:
    frames = cam.cylWarpFrames(frames, focalLength=focalLength,incrementAmount=cylWarpIncrement, cutFirstThenAppend=False)

# calculate initial homography
print("Calculating homography")
pt = time.time()
if origin2Stitch:
    homoList = cam.calcHomographyWarped2Origin(frames)
else:
    homoList = cam.calcHomographyWarped(frames)
etH = time.time() - pt
print(" - Elapsed time: ",etH)

if homoList is None:
   homoList = cam.openHomographyFile(f"Cylindrical{numCams}_Backup.npy")
   print("loaded backup")

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

    if origin2Stitch:
        frames = cam.cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=True, borderOnFirstAndFourth=True)
    else:
        frames = cam.cylWarpFrames(frames, focalLength=focalLength, incrementAmount=cylWarpIncrement, cutFirstThenAppend=False)

    times.append(time.time()-pt)


    # stitch frames
    pt = time.time()
    if origin2Stitch:
        pano = cam.stitchWarped2Origin(frames, homoList)
    else:
        pano = cam.stitchWarped(frames, homoList)
 
    times.append(time.time()-pt)


    # doesn't seem to work, works in cylStitchTest3 though, weird
    # # remove lines with hori kernel (boxFilter)
    # pt = time.time()

    # pano = cv.boxFilter(pano, 3, (3,1))

    # times.append(time.time()-pt)


    # calc frame rate
    per = time.time() - initTime
    frameRate = round(1/per,2)


    # round times
    for i, t in enumerate(times):
        times[i] = round(t,4)


    # Show pano, frame rate, individual times
    font = cv.FONT_HERSHEY_SIMPLEX
    size = .6
    x,y = 0,10#10, 500
    inc = 20
    # cv.putText(pano, f"Grab: {times[0]}s  Warp: {times[1]}s  Stitch: {times[2]}s", (100,80), font, 0.75,(0,0,255),2)
    stats = ["Grab frames:", "Warp frames:", "Stitch frames:"] #, "Blur:"]
    for i, text in enumerate(stats):
        cv.putText(pano, f"{text} {times[i]} s  ", (x,y+inc*i), font, size,(0,0,255),2)
    cv.putText(pano, f"Frame Rate: {frameRate} FPS", (x,y+inc*len(stats)), font, size,(0,0,255),2)
    cv.putText(pano, f"Homo Calc: {round(etH,4)} s", (x,y+inc*(len(stats)+1)), font, size,(0,0,255),2)
    cv.imshow("Pano",pano)


    # break on q
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()



    

