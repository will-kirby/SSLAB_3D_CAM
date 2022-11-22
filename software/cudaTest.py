
import time
import numpy as np
import cv2 as cv
from CameraSystemClass import CameraSystem

numCams=6
focalLength = 195
# imageIndex = [2,3,4,5,0,1]
imageIndex = range(6)
labNum=5

def _cylindricalWarp(img, K):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        start = time.time()
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

        print("Cylindrical coordinates calculations: "+str(time.time() - start))
        # img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...

        # warp the image according to cylindrical coords
        start = time.time()

        warpedImage = cv.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)

        print("Warped calculations: "+str(time.time() - start))

        # return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)#, borderMode=cv.BORDER_TRANSPARENT)
        return warpedImage

def cylWarpFrames(frames, focalLength = 200, incrementAmount = None, cutFirstThenAppend = False, borderOnFirstAndFourth=False):
        warpedFrames = []
        warpAmount = focalLength
        for frame in frames:
            h, w = frame.shape[:2]
            K = np.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]]) # mock intrinsics (normally from camera calibration?)
            
            # warp frame based on matrix K
            warpedFrames.append(_cylindricalWarp(frame, K))

            if incrementAmount:
                warpAmount -= incrementAmount #5 # attempting to adjust warp amount to be more as images increase, as with 

        # return [self._cylindricalWarp(frame, K) for frame in frames]
        """
        if borderOnFirstAndFourth:
            warpedFrames[1] = self.borderImg(warpedFrames[1])
            warpedFrames[4] = self.borderImg(warpedFrames[4])            

        # cut the first image in half, frame[0] is the right half, frame[-1] is the left half
        if cutFirstThenAppend:
            frame0L, warpedFrames[0] = self.cutImgVert(warpedFrames[0])
            warpedFrames.append(frame0L)
        """
        return warpedFrames

cam = CameraSystem([],compressCameraFeed=False)
frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],f"../images/lab_{labNum}/capture")

frames = cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=False)
