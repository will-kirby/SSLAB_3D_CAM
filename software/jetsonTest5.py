import numpy as np
import cv2 as cv
import imutils


cameras = []
for i in range(1,4):
    cameras.append(cv.VideoCapture(i))
    if not cameras[i-1].isOpened():
        print("Cannot open camera " + str(i))
        exit()
   
print(cameras)

stitcher = cv.Stitcher_create()

    
#while True:
    # RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # np.flip(image, axis=-1)
    
frames = []
for camera in cameras:  
    ret, frame = camera.read()
    frames.append(frame)
    
# index = 0
# for frame in frames:
#     cv.imshow('frame'+str(index), frame)
#     index = index + 1


# stitcher = cv.Stitcher_create()  # if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(frames)

if status == 0:
    cv.imshow('stitched',stitched)
else:
    print('error stitching',status)

# if cv.waitKey(1) == ord('q'):
#     break


for camera in cameras:
    camera.release()
cv.destroyAllWindows()