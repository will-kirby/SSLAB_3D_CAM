import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class CameraSystem:
    cameras = []
    cameraIndexList = None
    homographyMatrix = None
    overlapAmount = 56
    homographyMatrix = []
    

    def __init__(self, cameraIndexList=[], compressCameraFeed=True):
        # cameraIndexList is either a range or a list of the camera indices

        self.cameraIndexList = list(cameraIndexList)

        for i in cameraIndexList:
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

    def captureCameraImages(self, newResizedDimensions = None, interpolationType = cv.INTER_LINEAR):
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
        
        return frames

    def writeFramesToFile(self, frames, baseName="cameraImage", appendIndex=True, extension="jpg"):
        for index, frame in enumerate(frames):
            fileName = f"{baseName}{index if appendIndex else ''}.{extension}"
            print(f"Writing image: {fileName}")
            cv.imwrite(fileName, frame)

    def readFramesFromFiles(self, fileNames, basePath=""):
        return [cv.imread(basePath + name) for name in fileNames]

    def reorderCams(newOrderIndexList):
        # pass the index of each camera in the new order -> if want 3rd index first, pass in [3, ..]
        if max(newOrderIndexList) >= len(self.cameras):
            raise Exception("Bad new order passed in, index out of bounds")
        self.cameras[:] = [self.cameras[i] for i in list(newOrderIndexList)]

    def _findKPandDesSingle(self, frame):
        sift = cv.SIFT_create()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        return sift.detectAndCompute(gray,None)
        
    def _findKPandDesMultiple(self, frames):
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

    def matchDescriptors(self, descriptors1, descriptors2):
        # Brute Force Matcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1,descriptors2,k=2)

        # Apply ratio test - using the 2 nearest neighbors, only add if the nearest is much better than the other neighbor
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append(m)
        return goodMatches

    def findHomographyFromMatched(self, goodMatches, kp1, kp2):
        MIN_MATCH_COUNT = 10 # minimum of 10 matches to start stitching
        if len(goodMatches)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        else:
            print( "Not enough matches are found - {}/{}".format(len(goodMatches), MIN_MATCH_COUNT) )
            matchesMask = None
            M = None

        return (M, matchesMask)

    def homographyStitch(self, img1, img2, H):
        dst = cv.warpPerspective(img1,H,((img1.shape[1] + img2.shape[1]), img2.shape[0])) # warp the first image
        dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #plop the second down
        return dst


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

    def saveHomographyToFile(self, homographyMat : [], fileName : str = "savedHomographyMatrix.npy"):
        """
        Pass in homography matrix, gets saved to file
        - homographyMat is a list of homography matrices
        - this list gets converted to numpy array then saved
        """
        np.save(fileName, np.array(homographyMat))

    def openHomographyFile(self, fileName : str = "savedHomographyMatrix.npy"):
        """
        Open a homography matrix file, returns a list of each homography matrix
        """
        return list(np.load(fileName))

    def displayFrameMatplotlib(self, frame):
        plt.imshow(frame)
        plt.show()

    def calibrateMatrix(self):
        frames = self.captureCameraImages()
        kp, des = self._findKPandDesMultiple(frames)
        goodMatches = self.matchDescriptors(des[0], des[1])
        H, matchesMask = cam.findHomographyFromMatched(goodMatches, kp[0], kp[1])
        
        return H, matchesMask

    def fishEyeTransform(self, frame):
        # TBD. I think this is really important. If you apply fisheye to each image,
        # may just be able to shift the images together

        pass

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
            plt.imshow(img),plt.show()
            
        return img

if __name__ == "__main__":
    print("Constructing camera system")
    # cam = CameraSystem([0],False)
    # cam = CameraSystem([0,1],compressCameraFeed=False)
    cam = CameraSystem([],compressCameraFeed=False)


    # print("capturing images")
    # frames = cam.captureCameraImages()

    # print("Writing to file")
    # cam.writeFramesToFile(frames)

    print("Reading camera images")
    # frames = cam.readFramesFromFiles(["testImage1.jpg", "testImage0.jpg"],"code_testing/")
    frames = cam.readFramesFromFiles(["capture0.png", "capture1.png"],"functionality_testing/")

    kp, des = cam._findKPandDesMultiple(frames)
    goodMatches = cam.matchDescriptors(des[0], des[1])
    # cam.drawMatches(frames[0], kp[0], frames[1], kp[1], goodMatches)

    H, matchesMask = cam.findHomographyFromMatched(goodMatches, kp[0], kp[1])
    # cam.drawMatches(frames[0], kp[0], frames[1], kp[1], goodMatches, matchesMask)

    # cam.homographyStitch(frames[0], frames[1], H)

    # cam.overlapStitch(frames)

    # img3 = cv.drawKeypoints(frames[0], kp, None, color=(255,0,0))
    # cv.imwrite('image0KP.jpg', img3)