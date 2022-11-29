import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem
import time
import select
import sys

"""
    Currently uses pre-taken images for timing analysis.
"""

useStaticImages = True
origin2Stitch = True # if false uses left to right
numCams=6
focalLength = 146#190  146 
cylWarpIncrement = 6#5 6 ,
# imageIndex = [2,3,4,5,0,1]
imageIndex = range(6)
labNum=5
cutFirstThenAppend=True
homoList=None
cylMatrices=[]
NoStitchJustStack = False
regularFrames = []
Resized=None#(480,640)
num = 2
Path = f"../images/labPaper/{num}/image" #"f"../images/lab_{labNum}/capture"
filetype = ".jpg"
focalLengths =[]#[149, 140, 133, 128, 122, 119]##[210, 122, 150, 120, 120, 110, 105] #[108, 118, 130, 146, 132, 123] 


print("Starting timing anal prog")
# construct cam system and grab initial frames
if useStaticImages:
    cam = CameraSystem([],compressCameraFeed=False)
    frames = cam.readFramesFromFiles([str(n) + filetype for n in imageIndex], Path)
    cv.imshow("regular", np.hstack(frames))
else:
    cam = CameraSystem(range(numCams), useLinuxCam=True)
    frames = cam.captureCameraImages(Resized)
    #frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],f"../images/lab_{labNum}/capture")

if (not cutFirstThenAppend) and origin2Stitch:
    #cut and append before cylindrical warping
    frame0L, frames[0] = cam.cutImgVert(frames[0])
    frames.append(frame0L)

#compute the cylindrical coordinate transforms for the cameras

cylMatrices = cam.getCylCoords(frames, focalLength=focalLength, incrementAmount=cylWarpIncrement, focalLengths = focalLengths) #cylMatrices = cam.getCylCoords(frames, focalLength=focalLength, incrementAmount=cylWarpIncrement, focalLengths = focalLengths)

# warp frames

if origin2Stitch:
    #frames = cam.cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=True, borderOnFirstAndFourth=True)
    frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend = cutFirstThenAppend, borderOnFirstAndFourth=True)
else:
    #frames = cam.cylWarpFrames(frames, focalLength=focalLength,incrementAmount=cylWarpIncrement, cutFirstThenAppend=cutFirstThenAppend)
    frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend=cutFirstThenAppend)
    

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
   if origin2Stitch:
         homoList = cam.openHomographyFile("homography2origin.npy", "homography2origin_Backup.npy")
   else:
         homoList = cam.openHomographyFile("homographyLtoR.npy", f"Cylindrical{numCams}_Backup.npy")
   print("loaded backup")

while True:
    times = []
    initTime = time.time()
    if(select.select([sys.stdin,],[],[],0)[0] and sys.stdin.read(1) == 'r'): 
        pt = time.time()
        if origin2Stitch:
          homoList = cam.calcHomographyWarped2Origin(frames)
        else:
          homoList = cam.calcHomographyWarped(frames)


        etH = time.time() - pt
        print(" - Elapsed time: ",etH)

        if homoList is None:
           if origin2Stitch:
             homoList = cam.openHomographyFile("homography2origin.npy", "homography2origin_Backup.npy")
           else:
             homoList = cam.openHomographyFile(f"Cylindrical{numCams}_Backup.npy")
           print("loaded backup")
    pt = time.time()
    # grab frames
    if useStaticImages:
        frames = cam.readFramesFromFiles([str(n) + filetype for n in imageIndex],Path)
    else:
        frames = cam.captureCameraImages(Resized)
        regularFrames=frames
        cv.imshow("regular", np.hstack(regularFrames))
    times.append(time.time()-pt)

    if (not cutFirstThenAppend) and origin2Stitch:
       #cut and append before cylindrical warping
       frame0L, frames[0] = cam.cutImgVert(frames[0])
       frames.append(frame0L)
    
    # warp frames
    pt = time.time()

    if origin2Stitch:
        #frames = cam.cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=True, borderOnFirstAndFourth=True)
        frames = cam.applyCylWarp(frames, cylMatrices,cutFirstThenAppend = cutFirstThenAppend, borderOnFirstAndFourth=True)
    else:
        #frames = cam.cylWarpFrames(frames, focalLength=focalLength, incrementAmount=cylWarpIncrement, cutFirstThenAppend=cutFirstThenAppend)
        frames = cam.applyCylWarp(frames, cylMatrices, cutFirstThenAppend=cutFirstThenAppend)

    times.append(time.time()-pt)


    # stitch frames
    pt = time.time()
    cylStack = np.hstack(frames)
    if NoStitchJustStack:
       pano = cylStack
    else:
      cv.imshow("cylStack",cylStack)
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



    

