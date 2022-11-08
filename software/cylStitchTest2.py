import cv2 as cv
import sys
import numpy as np
from CameraSystemClass import CameraSystem, showTwoImgs

def borderMiddleImg(imgMiddle, borderAmount = None):
    # add border to left and right of middle so the left image has space to appear 
        # - this is necessary to line up the coordinates, as origin is at top left
    if borderAmount is None:
        borderAmount = imgMiddle.shape[1]
    return cv.copyMakeBorder(imgMain,0,0,borderAmount,borderAmount,cv.BORDER_CONSTANT)

def calcHomo(imgMain, imgToBeWarped):
    kp, des = cam._findKPandDesMultiple([imgToBeWarped, imgMain])
    goodMatches = cam.matchDescriptors(des[0], des[1])
    H, matchesMask = cam.findHomographyFromMatched(goodMatches, kp[0], kp[1])

    return H

def stitchSingle(imgMain, imgRight, H):
    h,w = imgMain.shape[:2]

    # warp right
    dstR = cv.warpPerspective(imgRight,H,(w, h))

    # get inverse mask of main
    imgMainGray = cv.cvtColor(imgMain,cv.COLOR_BGR2GRAY)
    ret, mainMaskInv = cv.threshold(imgMainGray, 0, 255, cv.THRESH_BINARY_INV)

    # apply mask to warped image (mask needs to be same size as input img)
    dstRMasked = cv.bitwise_and(dstR, dstR, mask=mainMaskInv)

    # add main in
    return cv.add(dstRMasked, imgMain)



def calcHomographyThree(imgLeft, imgMain, imgRight):
    """
    imgMiddle needs to have a border before calling this, use borderMiddleImg function

    returns Hl, Hr
    """

    # find keypoints for all images
    kp, des = cam._findKPandDesMultiple([imgLeft, imgMain, imgRight])

    # match keypoints for left and middle, as well as middle and right
    goodMatchesLeft = cam.matchDescriptors(des[0], des[1]) # the first arg is the image that is being warped, so left here
    goodMatchesRight = cam.matchDescriptors(des[2], des[1]) # and right img here

    # find the homography matrices to align the keypoints
    Hl, _ = cam.findHomographyFromMatched(goodMatchesLeft, kp[0], kp[1])
    Hr, _ = cam.findHomographyFromMatched(goodMatchesRight, kp[2], kp[1])

    return Hl, Hr

def stitchThree(imgLeft, imgMiddle, imgRight, Hl, Hr):
    """
    imgMiddle needs to have a border before calling this, use borderMiddleImg function

    returns a pano of all three images
    """
    h,w = imgMiddle.shape[:2]

    # warp left and right images
    dstR = cv.warpPerspective(imgRight,Hr,(w, h))
    dstL = cv.warpPerspective(imgLeft,Hl,(w, h))

    # additional step - erase the opposite edge
    # - if the warpPerspective goes to far, it could wrap around to the other side
    # - this would cause the right and left image to overlap in the edge region 
    #   -> makes it bright when adding together
    # - not an issue if cylindrical warp, as it all fits without wrap around

    # add the warped images together
    rAndL = cv.add(dstR, dstL)

    # get inverse mask of middle
    imgMiddleGray = cv.cvtColor(imgMiddle,cv.COLOR_BGR2GRAY)
    ret, middleMaskInv = cv.threshold(imgMiddleGray, 0, 255, cv.THRESH_BINARY_INV)

    # apply mask to l/r image
    rAndLMasked = cv.bitwise_and(rAndL, rAndL, mask=middleMaskInv)

    # add middle in
    return cv.add(rAndLMasked, imgMiddle)

def cropImageOLD(img):
    src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, imgMask = cv.threshold(src_gray, 0, 255, cv.THRESH_BINARY)

    # detect edges
    threshold = 150
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    # Find contours
    _, contours = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print(contours)
    cv.drawContours(img, contours, 0, (0,255,0),5)


    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # for i, c in enumerate(contours):
    #     contours_poly = cv.approxPolyDP(c, 3, True)
    #     boundRect = cv.boundingRect(c)
        
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #     cv.drawContours(drawing, contours_poly, i, color)
    #     cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
    #       (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    #     cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv.imshow('Contours', img)
    




imageIndex = [2,3,4,5,0,1]
cam = CameraSystem([],compressCameraFeed=False)
frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],"../images/lab_5/capture")
frames = cam.cylWarpFrames(frames, 197)

h,w = frames[0].shape[:2]
imgLeft = frames[0]
imgMain = frames[1]
imgRight = frames[2]

imgMiddleBorder = borderMiddleImg(imgMain)
Hl, Hr = calcHomographyThree(imgLeft, imgMiddleBorder, imgRight)
pano = stitchThree(imgLeft, imgMiddleBorder, imgRight, Hl, Hr)

cv.imshow("pano", pano)

panoCropped = cam.cropToBlob(pano)
cv.imshow("panoCropped", panoCropped)
cv.waitKey(0)


imgLeft = frames[3]
imgMain = frames[4]
imgRight = frames[5]

imgMiddleBorder = borderMiddleImg(imgMain)
Hl, Hr = calcHomographyThree(imgLeft, imgMiddleBorder, imgRight)
pano2 = stitchThree(imgLeft, imgMiddleBorder, imgRight, Hl, Hr)

# pano2 = np.roll(pano2, -200, axis=1)
# shiftAmount = 300
# w = pano2.shape[1]
# temp = np.zeros_like(pano2)
# temp[:,:w-shiftAmount,:] = pano2[:,shiftAmount:,:]
# pano2 = temp

# cv.imshow("pano3",pano2)
# cv.waitKey(0)
print("Pano shape", pano.shape)
# pano = cv.copyMakeBorder(pano,0,0,borderAmount,borderAmount,cv.BORDER_CONSTANT)
# pano = np.roll(pano, -350, axis=1)

print("Pano shape2", pano.shape)

# pano = cam.cropPanoImage(pano)
# pano2 = cam.cropPanoImage(pano2)

borderAmount = 1000
pano = cv.copyMakeBorder(pano,0,0,borderAmount,borderAmount,cv.BORDER_CONSTANT)


H = calcHomo(pano, pano2)
print(H)
pano3 = stitchSingle(pano, pano2, H)

pano3 = cam.cropPanoImage(pano3)



cv.imshow("pano",pano)
cv.imshow("pano2",pano2)
cv.imshow("pano3",pano3)

cam.displayFrameMatplotlib(pano3)
# cropImage(pano)

cv.waitKey(0)



# frames2 = cam.readFramesFromFiles([str(n) + ".png" for n in [2,1,0]],"../images/lab_5/capture")
# frames2 = [np.flip(frame,1) for frame in frames2]

# print("Successfuly opended frames, init shape:",frames[0].shape)
# print("Warping frames")
# frames = cam.cylWarpFrames(frames)
# frames2 = cam.cylWarpFrames(frames2)
# # print("Shape compare for center img:", frames[0].shape, frames)
# # showTwoImgs(frames[0], frames2[0])
# # cv.imshow("warpedImages",np.hstack((frames)))

# # cv.waitKey(0)

# print("Calculating homography")
# homoList = cam.calcHomographyWarped(frames)
# homoList2 = cam.calcHomographyWarped(frames2)

# print("Homography matrix:",homoList)

# panoImg = cam.stitchWarped(frames, homoList)
# panoImg2 = cam.stitchWarped(frames2, homoList2)
# panoImg2 = np.flip(panoImg2,1)
# cv.imshow("panoImage",panoImg)
# cv.imshow("panoImage2",panoImg2)

# #h = shape0, w=shape1
# h = panoImg.shape[0]
# w = panoImg.shape[1] + panoImg2.shape[1]
# firstLim = w - (frames[0].shape[1])//2
# panoComb = np.zeros((h,w, 3), dtype="uint8")

# panoComb[0:h, 0:firstLim] = panoImg2
# panoComb[0:h, firstLim:] = panoImg

# cv.imshow("panoImageBIG",panoComb)



# print("Waiting on keypress to destroy windows")
# cv.waitKey(0)
# cv.destroyAllWindows()

