import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem, showTwoImgs

startIndex = 0
endIndex = 2
cam = CameraSystem([],compressCameraFeed=False)
frames = cam.readFramesFromFiles([str(n) + ".png" for n in [2,3,4,5]],"../images/lab_5/capture")
frames2 = cam.readFramesFromFiles([str(n) + ".png" for n in [2,1,0]],"../images/lab_5/capture")
frames2 = [np.flip(frame,1) for frame in frames2]

print("Successfuly opended frames, init shape:",frames[0].shape)
print("Warping frames")
frames = cam.cylWarpFrames(frames)
frames2 = cam.cylWarpFrames(frames2)
# print("Shape compare for center img:", frames[0].shape, frames)
# showTwoImgs(frames[0], frames2[0])
# cv.imshow("warpedImages",np.hstack((frames)))

# cv.waitKey(0)

print("Calculating homography")
homoList = cam.calcHomographyWarped(frames)
homoList2 = cam.calcHomographyWarped(frames2)

print("Homography matrix:",homoList)

panoImg = cam.stitchWarped(frames, homoList)
panoImg2 = cam.stitchWarped(frames2, homoList2)
panoImg2 = np.flip(panoImg2,1)
cv.imshow("panoImage",panoImg)
cv.imshow("panoImage2",panoImg2)

#h = shape0, w=shape1
h = panoImg.shape[0]
w = panoImg.shape[1] + panoImg2.shape[1]
firstLim = w - (frames[0].shape[1])//2
panoComb = np.zeros((h,w, 3), dtype="uint8")

panoComb[0:h, 0:firstLim] = panoImg2
panoComb[0:h, firstLim:] = panoImg

cv.imshow("panoImageBIG",panoComb)



print("Waiting on keypress to destroy windows")
cv.waitKey(0)
cv.destroyAllWindows()

