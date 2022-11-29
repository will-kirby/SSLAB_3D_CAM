
import time
import numpy as np
import cupy as cp
import cv2 as cv
from CameraSystemClass import CameraSystem
from numba import jit
import statistics as stat

numCams=6
focalLength = 195
warpIncrement = 4
# imageIndex = [2,3,4,5,0,1]
imageIndex = range(6)
labNum=5

def _cylindricalWarp(img, K):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        #start = time.time()
        h_,w_ = img.shape[:2]

        #point = time.time()
        # pixel coordinates
        y_i, x_i = np.indices((h_,w_)) # does [[0,0,0...],[1,1,1...],[n-1,n-1,n-1...] then [0,1,2,3...n-1] * n
        X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog -> stacks [0,1,2,3] vert , then [0,0,0,], then [1,1,1,1]
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T # normalized coords (K^-1 * X^T)^T'
        #print("pixel coordinates npy: "+str(time.time() - point))
        #point = time.time()
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
        B = K.dot(A.T).T # project back to image-pixels plane
        #print("cylindrical coords npy: "+str(time.time() - point))

        #point = time.time()
        # back from homog coords
        
        B = B[:,:-1] / B[:,[-1]]
        #print("homogeneous coords npy: "+str(time.time() - point))
        
        #point = time.time()
        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        B = B.reshape(h_,w_,-1)

        #print("image bounds npy: "+str(time.time() - point))

        #print("Cylindrical coordinates npy calculations total: "+str(time.time() - start))
        # img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...

        # warp the image according to cylindrical coords
        """
        start = time.time()

        warpedImage = cv.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)

        print("Warped calculations npy: "+str(time.time() - start))
        """
        # return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)#, borderMode=cv.BORDER_TRANSPARENT)
        return B#warpedImage

@jit(nopython=True)
def _cylindricalWarp_numba(img, K, x_i, y_i):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        # start = time.time()
        h_,w_ = img.shape[:2]
       # point = time.time()
        # pixel coordinates
        #y_i, x_i = np.indices((h_,w_)) # does [[0,0,0...],[1,1,1...],[n-1,n-1,n-1...] then [0,1,2,3...n-1] * n
        X = np.stack((x_i,y_i,np.ones_like(x_i)),axis=-1).reshape(h_*w_,3) # to homog -> stacks [0,1,2,3] vert , then [0,0,0,], then [1,1,1,1]
        
        Kinv = np.linalg.inv(K)
        
        X = X.astype(np.float64)
        X = Kinv.dot(X.T).T # normalized coords (K^-1 * X^T)^T'
       # print("pixel coordinates npy: "+str(time.time() - point))
       
       # point = time.time()
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack((np.sin(X[:,0]),X[:,1],np.cos(X[:,0])),axis=-1).reshape(w_*h_,3)
        B = K.dot(A.T).T # project back to image-pixels plane
      #  print("cylindrical coords npy: "+str(time.time() - point))

       # point = time.time()
       # # back from homog coords
        B = B[:,:-1] / (B[:,-1].reshape(len(B[:,-1]), 1))
        #print("homogeneous coords npy: "+str(time.time() - point))
        
        #point = time.time()
        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        B = B.reshape(h_,w_,-1)

        #print("image bounds npy: "+str(time.time() - point))

        

        #print("Cylindrical coordinates npy calculations total: "+str(time.time() - start))
        # img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...

        # warp the image according to cylindrical coords
           
        
        #start = time.time()

        #warpedImage = cv.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)

        #print("Warped calculations npy: "+str(time.time() - start))
        
        # return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)#, borderMode=cv.BORDER_TRANSPARENT)
        return B#warpedImage


def _cylindricalWarp_CuPy(img, K):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        #start = time.time()
        h_,w_ = img.shape[:2]
        
        #point = time.time()
        # pixel coordinates
        y_i, x_i = cp.indices((h_,w_)) # does [[0,0,0...],[1,1,1...],[n-1,n-1,n-1...] then [0,1,2,3...n-1] * n
        #cp.cuda.Stream.null.synchronize()
        X = cp.stack([x_i,y_i,cp.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog -> stacks [0,1,2,3] vert , then [0,0,0,], then [1,1,1,1]
        #cp.cuda.Stream.null.synchronize()
        Kinv = cp.linalg.inv(K) 
        #cp.cuda.Stream.null.synchronize()
        X = Kinv.dot(X.T).T # normalized coords (K^-1 * X^T)^T
        #cp.cuda.Stream.null.synchronize()
        #print("pixel coordinates cupy: "+str(time.time() - point))

        #point = time.time()
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = cp.stack([cp.sin(X[:,0]),X[:,1],cp.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
        #cp.cuda.Stream.null.synchronize()
        B = K.dot(A.T).T # project back to image-pixels plane
        #cp.cuda.Stream.null.synchronize()
        #print("cylindrical coords cupy: "+str(time.time() - point))

        # back from homog coords
        #point = time.time()      
        B = B[:,:-1] / B[:,[-1]]
        #cp.cuda.Stream.null.synchronize()
        #print("homogeneous coords cupy: "+str(time.time() - point))
        
        #point = time.time()
        # make sure warp coords only within image bounds
        B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
        #cp.cuda.Stream.null.synchronize()
        B = B.reshape(h_,w_,-1)
        #cp.cuda.Stream.null.synchronize()
        #print("image bounds cupy: "+str(time.time() - point))
        

        #print("Cylindrical coordinates calculations cupy total: "+str(time.time() - start))
        # img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...

        # warp the image according to cylindrical coords
        B = cp.asnumpy(B)
        #cp.cuda.Stream.null.synchronize()
        #start = time.time()
       
        #warpedImage = cv.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)

        #print("Warped calculations cupy: "+str(time.time() - start))

        # return cv.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)#, borderMode=cv.BORDER_TRANSPARENT)
        return B#warpedImage

def cylWarpFrames(frames, focalLength = 200, incrementAmount = None, cutFirstThenAppend = False, borderOnFirstAndFourth=False):
        warpedFrames = []
        warpAmount = focalLength
        for frame in frames:
           # mock intrinsics (normally from camera calibration?)
            h, w = frame.shape[:2]
            K = np.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]])
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

def cylWarpFrames_CuPy(frames, focalLength = 200, incrementAmount = None, cutFirstThenAppend = False, borderOnFirstAndFourth=False):
        warpedFrames = []
        warpAmount = focalLength
        for frame in frames:
            h, w = frame.shape[:2]
            K = cp.array([[warpAmount,0,w/2],[0,warpAmount,h/2],[0,0,1]]) # mock intrinsics (normally from camera calibration?)
            
            # warp frame based on matrix K
            warpedFrames.append(_cylindricalWarp_CuPy(frame, K))

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

print("Initializing camera system...")
cam = CameraSystem([],compressCameraFeed=False)
print("Reading Frames...") 
frames = cam.readFramesFromFiles([str(n) + ".png" for n in imageIndex],f"../images/lab_{labNum}/capture")
frame = frames[0]
h, w = frame.shape[:2]
#frame = cp.asarray(frame)

#print("Applying cylindrical warp with cp...")


K = cp.array([[focalLength,0,w/2],[0,focalLength,h/2],[0,0,1]])
K1 = np.array([[focalLength,0,w/2],[0,focalLength,h/2],[0,0,1]])


y_i, x_i = np.indices((h,w))
B=_cylindricalWarp_numba(frame, K1, x_i, y_i)
B1=_cylindricalWarp(frame, K1)
print(np.allclose(B, B1))

"""
warpedImage = cv.remap(frame, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)
cv.imshow("Warped Frame", warpedImage)
cv.waitKey(5000)
warpedImage = cv.remap(frame, B1[:,:,0].astype(np.float32), B1[:,:,1].astype(np.float32), cv.INTER_AREA)
cv.imshow("Warped Frame", warpedImage)
cv.waitKey(5000)
"""


"""
warpedFrame=_cylindricalWarp_CuPy(frame, K)
warpedFrame=_cylindricalWarp_CuPy(frame, K)

"""

"""
times_reg = []
for i in range(1000):
   start = time.time()
   warpedFrame=_cylindricalWarp(frame, K1)
   end = time.time()
   times_reg.append(end-start)
   #print(f"Warped calculations regular {i}: {end - start}")

print(f"Max regular numpy time: {max(times_reg)}, Min regular numpy time: {min(times_reg)}, Average regular numpy time: {stat.mean(times_reg)}, Median regular numpy time: {stat.median(times_reg)}")


y_i, x_i = np.indices((h,w))


times_numba = []
for i in range(1000):
   start = time.time()
   warpedFrame=_cylindricalWarp_numba(frame, K1, x_i, y_i)
   end = time.time()
   times_numba.append(end-start)
   #print(f"Warped calculations numba {i}: {end - start}")

print(f"Max numba time: {max(times_numba)}, Min numba time: {min(times_numba)}, Average numba time: {stat.mean(times_numba)}, Median numba time: {stat.median(times_numba)}")

times_cupy = []
for i in range(1000):
   start = time.time()
   warpedFrame=_cylindricalWarp_CuPy(frame, K)
   end = time.time()
   times_cupy.append(end-start)
   #print(f"Warped calculations numba {i}: {end - start}")

print(f"Max cupy time: {max(times_cupy)}, Min cupy time: {min(times_cupy)}, Average cupy time: {stat.mean(times_cupy)}, Median cupy time: {stat.median(times_cupy)}")
"""


#warpedFrame=_cylindricalWarp_CuPy(frame, K)

"""
start = time.time()
h_,w_ = frame.shape[:2]


point = time.time()
# pixel coordinates
y_i, x_i = cp.indices((h_,w_)) # does [[0,0,0...],[1,1,1...],[n-1,n-1,n-1...] then [0,1,2,3...n-1] * n
X = cp.stack([x_i,y_i,cp.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog -> stacks [0,1,2,3] vert , then [0,0,0,], then [1,1,1,1]
Kinv = cp.linalg.inv(K) 
X = Kinv.dot(X.T).T # normalized coords (K^-1 * X^T)^T
print("pixel coordinates cupy: "+str(time.time() - point))

point = time.time()
# calculate cylindrical coords (sin\theta, h, cos\theta)
A = cp.stack([cp.sin(X[:,0]),X[:,1],cp.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
B = K.dot(A.T).T # project back to image-pixels plane
print("cylindrical coords cupy: "+str(time.time() - point))

# back from homog coords
point = time.time()
B = B[:,:-1] / B[:,[-1]]
print("homogeneous coords cupy: "+str(time.time() - point))

point = time.time()
# make sure warp coords only within image bounds
B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
B = B.reshape(h_,w_,-1)
print("image bounds cupy: "+str(time.time() - point))


print("Cylindrical coordinates calculations cupy total: "+str(time.time() - start))
# img_rgba = cv.cvtColor(img,cv.COLOR_BGR2BGRA) # for transparent borders...

# warp the image according to cylindrical coords
B = cp.asnumpy(B)
start = time.time()

warpedImage = cv.remap(frame, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv.INTER_AREA)

print("Warped calculations cupy: "+str(time.time() - start))
"""

#cv.imshow("Warped Frame", warpedFrame)
#cv.waitKey(0)

#print(cp.cuda.runtime.getDeviceCount())




#print("Applying cylindrical warps with np...")
#frames = cylWarpFrames(frames, focalLength=focalLength, cutFirstThenAppend=False)

#print("Applying cylindrical warps with cp...")
#frames = cylWarpFrames_CuPy(frames, focalLength=focalLength, cutFirstThenAppend=False)


print("finished")

"""
s = time.time()
x_gpu = cp.ones(10)

cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)

s = time.time()
x_gpu *= 5
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)

print("initial calls done")

### Numpy and CPU
s = time.time()
x_cpu = np.ones(10000000)
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones(10000000)
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)

s = time.time()
x_cpu *= 5
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu *= 5
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)
"""
