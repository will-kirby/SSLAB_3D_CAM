import cv2 as cv
import numpy as np

class CameraSystem:
    cameras = []
    cameraIndexList = None
    homographyMatrix = None
    

    def __init__(self, cameraIndexList, compressCameraFeed=True):
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
        return [cv.imread(name) for name in fileNames]

    def calculateHomographyMatrices(self):
        frames = self.captureCameraImages()

    def _framesToGrayScale(self, frames):
        return [cv.cvtColor(frame,cv.COLOR_BGR2GRAY) for frame in frames]

    def _findKPandDesSingle(self, frame):
        sift = cv.SIFT_create()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        return sift.detectAndCompute(gray,None)
        
    def _findKPandDesMultiple(self, frames):
        sift = cv.SIFT_create()
        grays = self._framesToGrayScale(frames)
        framesKpsList = []
        framesDescripList = []
        for grayFrame in grays:
            kps, des = sift.detectAndCompute(grayFrame,None)
            # kps = np.float32([kp.pt for kp in kps])
            
            framesKpsList.append(kps)
            framesDescripList.append(des)

        return (framesKpsList, framesDescripList)

    def _matchDescriptorsWithFilter(descriptors1, descriptors2):
        # Brute Force Matcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(descriptors1,descriptors2,k=2)

        # Apply ratio test - using the 2 nearest neighbors, only add if the nearest is much better than the other neighbor
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append([m])


    def stitchCameras(self, stitchType):
        pass

if __name__ == "__main__":
    print("Constructing camera system")
    # cam = CameraSystem([0],False)
    cam = CameraSystem([],compressCameraFeed=False)

    # print("capturing images")
    # frames = cam.captureCameraImages()

    # print("Writing to file")
    # cam.writeFramesToFile(frames)

    print("Reading camera images")
    frames = cam.readFramesFromFiles(["software/code_testing/testImage0.jpg"])
    kp, des = cam._findKPandDesSingle(frames[0])

    img3 = cv.drawKeypoints(frames[0], kp, None, color=(255,0,0))
    cv.imwrite('image0KP.jpg', img3)