import cv2 as cv
import numpy as np
from CameraSystemClass import CameraSystem

focalLength = 190
cylWarpIncrement = 3


# imageIndex = [2,3,4,5,0,1]
imageIndex = range(6)
labNum=5
num=1
Path = f"../images/lab_{labNum}/capture" #f"../images/labPaper/{num}/image"
cam = CameraSystem([],compressCameraFeed=False)
frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],Path)
for i,frame in enumerate(frames):
  if frame is None:
     print(f"Failed to read frame {i}")
     exit()
frames = cam.cylWarpFrames(frames, focalLength=focalLength, incrementAmount=cylWarpIncrement, cutFirstThenAppend=True, borderOnFirstAndFourth=True)

HExtra = cam.calcHomo(frames[-2], frames[-1])

frames[-2] = cam.stitchSingle(frames[-2], frames[-1], HExtra)
cv.imshow("Right End",frames[-2])
cv.waitKey(0)
HlL, HrL = cam.calcHomographyThree(frames[0], frames[1], frames[2])
HlR, HrR = cam.calcHomographyThree(frames[3], frames[4], frames[5])

panoL = cam.stitchThree(frames[0], frames[1], frames[2], HlL, HrL)
panoR = cam.stitchThree(frames[3], frames[4], frames[5], HlR, HrR)

"""""
HExtra = cam.calcHomo(panoR, frames[-1])

panoR = cam.stitchSingle(panoR, frames[-1], HExtra)
cv.imshow("Right End",panoR)
"""


# cv.imwrite(f"testImages/TestPanoLeft{focalLength}_Lab{labNum}.jpg",panoL)
# cv.imwrite(f"testImages/TestPanoRight{focalLength}_Lab{labNum}.jpg",panoR)

HFinal = cam.calcHomo(panoL, panoR)

pano = cam.stitchSingle(panoL, panoR, HFinal)
pano = cam.cropToBlob(pano)
cv.imshow("Pano", pano)
pano = cv.boxFilter(pano, 3, (3,1))
#cv.imshow("Pano Blurred", pano)
cv.waitKey(0)
#cv.imwrite(f"testImages/TestPanoAll{focalLength}_Lab{labNum}_testBlur.jpg",pano)









