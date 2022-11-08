import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class CameraSystem:
    cameras = []
    cameraIndexList = None
    homographyMatrix = None
    overlapAmount = 56
    homographyMatrix = []
    warpAmount = 200
    
    def __init__(self, cameraIndexList=[], compressCameraFeed=True, useLinuxCam=True):
        # cameraIndexList is either a range or a list of the camera indices

        self.cameraIndexList = list(cameraIndexList)

        for i in cameraIndexList:
            if useLinuxCam:
                camera = cv.VideoCapture(f"/dev/camera{i}")
            else:
                camera = cv.VideoCapture(i)

            if not camera.isOpened():
                print(f"Cannot open camera {i}")
                raise Exception(f"Unable to open camera {i}")

            if compressCameraFeed:
                camera.set(cv.CAP_PROP_FPS, 15)
                camera.set(cv.CAP_PROP_FRAME_WIDTH, 320)
                camera.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
                camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')) # what does this do?

            self.cameras.append(camera)

    def __del__(self):
        for camera in self.cameras:
            camera.release()

# Captures images

    def captureCameraImages(self, cylWarp = False, newResizedDimensions = None, interpolationType = cv.INTER_LINEAR):
        # newDimensions is a tuple - (newheight, newWidth)
        # interpolationType is only used if newDimensions is set
        # returns None if error, camera frames otherwise

        frames = []
        for i, camera in enumerate(self.cameras):
            ret, frame = camera.read()
            if not ret:
                print(f"Can't receive frame from camera {i}")
                return None

            if newResizedDimensions and len(newResizedDimensions) == 2:
                frame = cv.resize(frame, dsize=newResizedDimensions, interpolation=interpolationType)

            frames.append(frame)

        if cylWarp:
            frames = self.cylWarpFrames(frames)
    
        return frames

# save/load util functions

    def saveHomographyToFile(self, homographyMat : [], fileName : str = "savedHomographyMatrix.npy"):
        """
        Pass in homography matrix, gets saved to file
        - homographyMat is a list of homography matrices
        - this list gets converted to numpy array then saved
        """
        np.save(fileName, np.array(homographyMat))

    def openHomographyFile(self, fileName : str = "homographyMatrix.npy", backupFile : str = "savedHomographyMatrix.npy"):
        """
        Open a homography matrix file, returns a list of each homography matrix
        Also takes in a backup file, falls back on it if unable to open the standard
        """
        try:
            matrix = list(np.load(fileName))
        except:
            print(f"Unable to open {fileName}, falling back on backup")
            matrix = list(np.load(backupFile))

        return matrix
        
    def writeFramesToFile(self, frames, baseName="cameraImage", appendIndex=True, extension="jpg"):
        for index, frame in enumerate(frames):
            fileName = f"{baseName}{index if appendIndex else ''}.{extension}"
            print(f"Writing image: {fileName}")
            cv.imwrite(fileName, frame)

    def readFramesFromFiles(self, fileNames, basePath=""):
        return [cv.imread(basePath + name) for name in fileNames]

# DISPLAY (for debugging) functions

    def displayFrameMatplotlib(self, frame):
        plt.imshow(frame)
        plt.show()
 
    def drawMatches(self, img1, kp1, img2, kp2, goodMatches, matchesMask=None, displayImage=True):
        # if matchesMask is provided (from homography matrix step) then the matches will be
        # filtered down to more exact ones

        if matchesMask:
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img = cv.drawMatches(img1,kp1,img2,kp2,goodMatches,None,**draw_params)
        else:
            # cv.drawMatchesKnn expects list of lists as matches.
            goodMatches = [[m] for m in goodMatches]
            img = cv.drawMatchesKnn(img1,kp1,img2,kp2,goodMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        if displayImage:
            plt.figure()
            plt.imshow(img)
            plt.show()
            
        return img

# PREPROCESSING functions

    def cropToBlob(self, img, showImg=False):
        src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, imgMask = cv.threshold(src_gray, 0, 255, cv.THRESH_BINARY)

        x,y,w,h = cv.boundingRect(imgMask)
        x2, y2 = x+w, y+h

        if showImg:
            rectImg = cv.rectangle(img.copy(), (x,y),(x2,y2), (255,0,0),4)
            cv.imshow('BoundingRect', rectImg)

        return img[y:y2, x:x2]

    def borderImg(self, img, borderAmount = None):
        # add border to left and right of middle so the left image has space to appear 
            # - this is necessary to line up the coordinates, as origin is at top left
        if borderAmount is None:
            borderAmount = img.shape[1]
        return cv.copyMakeBorder(img,0,0,borderAmount,borderAmount,cv.BORDER_CONSTANT)

    def _cylindricalWarp(self, img, K):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        
        h_,w_ = img.shape[:2]

        # pixel coordinates
        y_i, x_i = np.indices((h_,w_)) # does [[0,0,0...],[1,1,1...],[n-1,n-1,n-1...] then [0,1,2,3...n-1] * n
        X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog -> stacks [0,1,2,3] vert , then [0,0,0,], then [1,1,1,1]
        Kinv = np.linalg.inv(K) 
        X = Kinv.dot(X.T).T # normalized coords (K^-1 * X^T)^T

        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
        B = K.dot(A.T).T # project back to image-pixels plane

        # back from homog coords
        B = B[:,:-1] / B[:,[-1]]

        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        B = B.reshape(h_,w_,-1)
        # img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...

        # warp the image according to cylindrical coords
        
        # return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA, borderMode=cv.BORDER_TRANSPARENT)
        return cv.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA) #, borderMode=cv.BORDER_TRANSPARENT)

    def cylWarpFrames(self, frames, focalLength = 200, incrementAmount = None):
        warpedFrames = []
        # warpAmount = self.warpAmount
        warpAmount = focalLength
        for frame in frames:
            h, w = frame.shape[:2]
            K = np.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]]) # mock intrinsics (normally from camera calibration?)
            warpedFrames.append(self._cylindricalWarp(frame, K))

            if incrementAmount:
                warpAmount -= incrementAmount #5 # attempting to adjust warp amount to be more as images increase, as with 

        # return [self._cylindricalWarp(frame, K) for frame in frames]
        return warpedFrames

    def overlapImgs(self, imgLeft, imgRight):
        """
        imgLeft is the smaller dimension image -> will get resized to larger
        imgRight is getting imageLeft layed on top of it

        """

        # resize imageLeft -> is there an opencv function to do this faster?
        # something like resize but to add a black border instead of change the whole thing
        imgL = np.zeros_like(imgRight)
        imgL[0:imgLeft.shape[0], 0:imgLeft.shape[1]] = imgLeft

        maskRight = 1 - (np.sum(imgL,axis=-1) > 0) # true if value not (0,0,0)
        maskRight = np.dstack((maskRight, maskRight, maskRight)) # make one for each pixcel channel
        imgRightMasked = imgRight * maskRight
        
        overlapped = cv.addWeighted(imgL, 1, imgRightMasked, 1, 0, dtype=cv.CV_8U)

        return overlapped

# HOMOGRAPHY functions

    def _findKPandDesSingle(self, frame):
        """
            SIFT detection for key points, single image

            returns (kp's, descriptors)
        """

        sift = cv.SIFT_create()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        return sift.detectAndCompute(gray,None)
        
    def _findKPandDesMultiple(self, frames):
        """
            SIFT detection for key points, multiple images

            returns (kp's, descriptors)
        """

        sift = cv.SIFT_create()
        grays = [cv.cvtColor(frame,cv.COLOR_BGR2GRAY) for frame in frames]
        framesKpsList = []
        framesDescripList = []
        for grayFrame in grays:
            kps, des = sift.detectAndCompute(grayFrame,None)
            # kps = np.float32([kp.pt for kp in kps])
            
            framesKpsList.append(kps)
            framesDescripList.append(des)

        return (framesKpsList, framesDescripList)

    def _matchDescriptors(self, descriptors1, descriptors2):
        # Brute Force Matcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1,descriptors2,k=2)

        # Apply ratio test - using the 2 nearest neighbors, only add if the nearest is much better than the other neighbor
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append(m)
        return goodMatches

    def _findHomographyFromMatched(self, goodMatches, kp1, kp2, MIN_MATCH_COUNT=10):
        # MIN_MATCH_COUNT the minimum matches to start stitching

        if len(goodMatches)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            print(f"Matches found - {len(goodMatches)}")
        else:
            print(f"Not enough matches are found - {len(goodMatches)}/{MIN_MATCH_COUNT}")
            matchesMask = None
            M = None

        return (M, matchesMask)

    def calcHomo(self, imgMain, imgToBeWarped):
        """
        Takes in two images, warps the toBeWarped onto main
        """
        kp, des = self._findKPandDesMultiple([imgToBeWarped, imgMain])
        goodMatches = self._matchDescriptors(des[0], des[1])
        H, matchesMask = self._findHomographyFromMatched(goodMatches, kp[0], kp[1])

        return H

    def calcHomographyThree(self, imgLeft, imgMain, imgRight):
        """
        Intended for use with pre-warped images
        imgMiddle needs to have a border before calling this, use borderMiddleImg function

        returns Hl, Hr
        """

        # find keypoints for all images
        kp, des = self._findKPandDesMultiple([imgLeft, imgMain, imgRight])

        # match keypoints for left and middle, as well as middle and right
        goodMatchesLeft = self._matchDescriptors(des[0], des[1]) # the first arg is the image that is being warped, so left here
        goodMatchesRight = self._matchDescriptors(des[2], des[1]) # and right img here

        # find the homography matrices to align the keypoints
        Hl, _ = self._findHomographyFromMatched(goodMatchesLeft, kp[0], kp[1])
        Hr, _ = self._findHomographyFromMatched(goodMatchesRight, kp[2], kp[1])

        return Hl, Hr

    # homography for left to right stitching
    def calcHomographyWarped(self, frames, saveHomo = True, fileName="fallbackHomography.npy"):
        """ 
        Pass in cyl-warped frames. Ends up stitching them together to find keypoints and calculate homography matrices
        returns a list of the homography matrices

        Accounting for failure:
            if keypoint matching fails, then stops, as it needs the previous result to keep going

            Possible alternative is to preload the default matrices, if one of the matches fails fall back on the default
            for that single failure?
        """

        panoImg = frames[0]
        homoList = []
        for i in range(1,len(frames)):

            # find key points
            kp, des = self._findKPandDesMultiple([frames[i], panoImg])
            goodMatches = self._matchDescriptors(des[0], des[1])
            H, matchesMask = self._findHomographyFromMatched(goodMatches, kp[0], kp[1])

            # use H to stitch, then loop and find keypoints again with larger image
            if H is not None:
                panoImg = self._stitchWarpedSegment(panoImg, frames[i], H)
            else:
                print(f"Error finding keypoints with image {i} and panoImage")
                return None
            homoList.append(H)

        if saveHomo:
            print(f"Saving homo Hleft, Hright to file {fileName}")
            self.saveHomographyToFile(homoList,fileName)

        return homoList

    def calibrateMatrix(self, save=False, filename: str = "testFlaskHomography.npy"):
        frames = self.captureCameraImages()
        kp, des = self._findKPandDesMultiple(frames)
        goodMatches = self._matchDescriptors(des[0], des[1])
        H, matchesMask = self._findHomographyFromMatched(goodMatches, kp[0], kp[1])

        if save:
            if H is not None:
                # Save homo to file
                print(f"Saving homo H to file {filename}")
                self.saveHomographyToFile([H],filename)

            else:
                print("Not enough matches detected to compute homography")

        return H, matchesMask

    # for use with normal images (not warped)
    def calibrateMatrixTriple(self, imgL, imgM, imgR, save=False, filename: str = "testFlaskHomography.npy"):
        # kp,des,and match: the parameters are the right image then left image, the right image then gets warped
        kp, des = self._findKPandDesMultiple([imgR, imgM, np.flip(imgL,1), np.flip(imgM,1)])
        goodMatchesRight = self._matchDescriptors(des[0], des[1])
        Hright, matchesMask = self._findHomographyFromMatched(goodMatchesRight, kp[0], kp[1])
        goodMatchesLeft = self._matchDescriptors(des[2], des[3])
        Hleft, matchesMask = self._findHomographyFromMatched(goodMatchesLeft, kp[2], kp[3])

        if save:
            if Hleft is not None and Hright is not None:
            # Save homo to file
                print(f"Saving homo Hleft, Hright to file {filename}")
                self.saveHomographyToFile([Hleft, Hright],filename)

            else:
                print("Not enough matches detected to compute homography")

        return Hleft, Hright

    # for use with normal images, not warped
    def calibrateMatrixTripleTwice(self, frames, save=False, filename: str = "testFlaskHomography.npy"):
        Hlf, Hrf = self.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        Hlb, Hrb = self.calibrateMatrixTriple(frames[3], frames[4], frames[5])

        if save:
            if Hlf is not None and Hrf is not None and Hlb is not None and Hrb is not None:
                print(f"Saving homo to file {filename}")
                self.saveHomographyToFile([Hlf, Hrf, Hlb, Hrb],filename)

            else:
                print("Not enough matches detected to compute homography")

        return Hlf, Hrf, Hlb, Hrb

# STITCHING functions

    def stitchSingle(self, imgMain, imgRight, H):
        """
            Stitching with prewarped images
        """
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

    def stitchThree(self, imgLeft, imgMiddle, imgRight, Hl, Hr):
        """
        Needs pre-warped images
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


    # The below two functions are used to stich left to right, adding segments on the right for pre-warped
    def _stitchWarpedSegment(self, mainPanoImg, newRightSegment, homographyMat):
        dst = cv.warpPerspective(newRightSegment,homographyMat,(mainPanoImg.shape[1] + newRightSegment.shape[1], newRightSegment.shape[0])) # warp the first image
        return self.overlapImgs(mainPanoImg, dst)

    def stitchWarped(self, frames, homographyList):
        """
        Takes in cyl-warped frames, Homography matricies 0-4 (5 matrices for 6 frames)
        returns a pano of the frames
        """
        stitchedImage = None
        if len(frames)-1 == len(homographyList):
            stitchedImage = frames[0]
            
            for i in range(1,len(frames)):
                stitchedImage = self._stitchWarpedSegment(stitchedImage, frames[i], homographyList[i-1])

        return stitchedImage

    # stitch single, for normal (non-warped) images
    def homographyStitch(self, img1, img2, H):
        dst = cv.warpPerspective(img1,H,((img1.shape[1] + img2.shape[1]), img2.shape[0])) # warp the first image
        dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #plop the second down
        return dst

    # stitch triple, for normal (non-warped) images
    def tripleHomographyStitch(self, imgL, imgM, imgR, Hl, Hr):
        dstRight = cv.warpPerspective(imgR,Hr,((imgM.shape[1] + imgR.shape[1]), imgR.shape[0])) # warp the first image
        dstRight[0:imgM.shape[0], 0:imgM.shape[1]] = imgM #plop the second down
        
        imgL = np.flip(imgL,1)
        imgM = np.flip(imgM,1)
        dstLeft = cv.warpPerspective(imgL,Hl,((imgM.shape[1] + imgL.shape[1]), imgL.shape[0])) # warp the first image
        dstLeft[0:imgM.shape[0], 0:imgM.shape[1]] = imgM #plop the second down
        dstLeft = np.flip(dstLeft,1)

        height = dstLeft.shape[0]
        width = imgL.shape[1] + imgM.shape[1] + imgR.shape[1]
        stitch = np.zeros((height, width, 3), dtype="uint8")
        stitch[0:height, 0:dstLeft.shape[1]] = dstLeft
        stitch[0:height, imgL.shape[1]:] = dstRight

        return stitch

    # stitch triplex2, for normal (non-warped) images
    def homographyStitchTripleTwice(self, frames, Hl, Hr, Hl2, Hr2):
        im1 = self.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
        im2 = self.tripleHomographyStitch(frames[3], frames[4], frames[5], Hl2, Hr2)

        return np.hstack((im1,im2))#np.vstack((im1,im2))

    # simple stich, just moves images until they overlap by a constant amount
    # a bit ambitious, as it doesn't really work
    def overlapStitch(self, frames, overlapAmount=None):
        if overlapAmount:
            shiftAmount = overlapAmount
        else:
            shiftAmount = self.overlapAmount
        overlappedImage = np.zeros((max([frame.shape[0] for frame in frames]), sum([frame.shape[1] for frame in frames]), 3), dtype="uint8" )
    
        currentWidth = 0
        for i, frame in enumerate(frames):
            overlappedImage[0:frame.shape[0], (currentWidth-shiftAmount*i):(currentWidth + frame.shape[1]-shiftAmount*i)] = frame
            currentWidth += (frame.shape[1] - shiftAmount)

        return overlappedImage



def showTwoImgs(img1, img2):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()


if __name__ == "__main__":
    
    print("Constructing camera system")
    # cam = CameraSystem([0],False)
    # cam = CameraSystem([0,1],compressCameraFeed=False)
    cam = CameraSystem([],compressCameraFeed=False)

    # img = cv.imread("functionality_testing/capture3.png")
    # if img is None:
    #     print("Bad read")
    #     exit() 

    frames = cam.readFramesFromFiles(["capture3.png", "capture4.png"],"../images/lab5/")

    img = frames[1]
    h, w = img.shape[:2]
    warpAmount = 200 #200 # this should probably come from camera calibration, but whatever
    K = np.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]]) # mock intrinsics (normally from camera calibration?)
    img_cyl1 = cam._cylindricalWarp(img, K)

    img = frames[0]
    h, w = img.shape[:2]
    K = np.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]]) # mock intrinsics (normally from camera calibration?)
    img_cyl0 = cam._cylindricalWarp(img, K)

    img_cyl0_1 = cam._cylindricalWarp(img_cyl0, K)

    showTwoImgs(img_cyl0, img_cyl0_1)

    warpAmount = 200 #200 # this should probably come from camera calibration, but whatever
    K = np.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]]) # mock intrinsics (normally from camera calibration?)
    img_cyl0_2 = cam._cylindricalWarp(img, K)

    plt.figure()
    plt.subplot(131)
    plt.imshow(img_cyl0)

    plt.subplot(132)
    plt.imshow(img_cyl0_1)

    plt.subplot(133)
    plt.imshow(img_cyl0_2)
    plt.show()
    
    # kp, des = cam._findKPandDesMultiple([frames[0], img_cyl])
    # kp, des = cam._findKPandDesMultiple([frames[1], frames[0]])
    kp, des = cam._findKPandDesMultiple([img_cyl1, img_cyl0])
    goodMatches = cam._matchDescriptors(des[0], des[1])
    H, matchesMask = cam._findHomographyFromMatched(goodMatches, kp[0], kp[1])
    # print("Homo:",H)

    if H is not None:
        # dst = cam.homographyStitch(img_cyl1, img_cyl0, H)

        # plt.figure()
        # plt.subplot(121)
        
        dst = cv.warpPerspective(img_cyl1,H,((img_cyl0.shape[1] + img_cyl1.shape[1]), img_cyl1.shape[0])) # warp the first image
        # dst = cv.cvtColor(dst,cv.COLOR_BGR2BGRA)

        cam.overlapImgs(img_cyl0, dst)
        # plt.imshow(dst)

        # alphaMask = (np.sum(img_cyl0,axis=-1) > 0) * 255.0
        # print("Alpha mask shape",alphaMask.shape)
        # img_cyl0 = np.dstack((img_cyl0, alphaMask))

        
    
        # dst[0:img_cyl0.shape[0], 0:img_cyl0.shape[1]] = img_cyl0 #plop the second down
        # plt.subplot(122)
        # plt.imshow(dst)
        # # plt.show()

        # plt.figure()
        # plt.imshow(alphaMask)
        # plt.show()

        # result_1 = blend_non_transparent(face_img, overlay_img)

    else:
        print("H is none")



    # print("capturing images")
    # frames = cam.captureCameraImages()

    # print("Writing to file")
    # cam.writeFramesToFile(frames)

    # print("Reading camera images")
    # frames = cam.readFramesFromFiles(["testImage1.jpg", "testImage0.jpg"],"code_testing/")
    # frames = cam.readFramesFromFiles(["capture0.png", "capture1.png","capture2.png"],"functionality_testing/")
    # frames = cam.readFramesFromFiles(["capture3.png", "capture4.png"],"images/lab_5/")


    # triple pano
    # Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
    # dst = cam.tripleHomographyStitch(frames[0], frames[1], frames[2], Hl, Hr)
    # plt.imshow(dst)
    # plt.show()




    # kp, des = cam._findKPandDesMultiple(frames)

    # imageA = 2
    # imageB = 1
    # goodMatches = cam._matchDescriptors(des[imageA], des[imageB])
    # H, matchesMask = cam._findHomographyFromMatched(goodMatches, kp[imageA], kp[imageB])
    # # matched = cam.drawMatches(frames[imageA], kp[imageA], frames[imageB], kp[imageB], goodMatches, matchesMask, False)
    # # dst = cam.homographyStitch(frames[imageA], frames[imageB], H)
    # # plt.figure()
    # # plt.subplot(211)
    # # plt.imshow(matched)

    # # plt.subplot(212)
    # # plt.imshow(dst)
    # # plt.show()

    # imageA = 0
    # imageB = 1
    # frames = list(map(lambda ar : np.flip(ar,1), frames[0:2]))
    # frames = [np.flip(frames[0],1), np.flip(frames[1],1)]
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(frames[0])
    # plt.subplot(222)
    # plt.imshow(frames[1])

    # kp, des = cam._findKPandDesMultiple(frames)
    # goodMatches = cam._matchDescriptors(des[imageA], des[imageB])
    # H, matchesMask = cam._findHomographyFromMatched(goodMatches, kp[imageA], kp[imageB])
    # dst = cv.warpPerspective(frames[0],H,(640, 480)) # warp the first image
    # plt.subplot(212)

    # dst = cam.homographyStitch(frames[imageA],frames[imageB],H)
    # dst = np.flip(dst,1)
    # plt.figure()
    # plt.imshow(dst)
    # plt.show()

    # cv.imshow("thing",dst)
    # time.sleep(3)

    # cam.overlapStitch(frames)

    # img3 = cv.drawKeypoints(frames[0], kp, None, color=(255,0,0))
    # cv.imwrite('image0KP.jpg', img3)
