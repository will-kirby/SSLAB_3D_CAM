# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2 as cv
import imutils


cameras = []
for i in range(0,4):
    cameras.append(cv.VideoCapture('/dev/video' + str(i)))
    if not cameras[i].isOpened():
        print("Cannot open camera " + str(i))
        exit()

# cap0 = cv.VideoCapture('/dev/video0')
# cap1 = cv.VideoCapture('/dev/video1')
# cap2 = cv.VideoCapture('/dev/video2')
# cap3 = cv.VideoCapture('dev/video3')


# if not cap0.isOpened():
#     print("Cannot open camera 0")
#     exit()

# if not cap1.isOpened():
#     print("Cannot open camera 1")
#     exit()

# if not cap2.isOpened():
#     print("Cannot open camera 2")
#     exit()
index = 0
for camera in cameras:
    ret, frame =  camera.read()
    cv.imwrite('capture' + str(index) + '.png', frame)
    camera.release()

imagePaths = ['capture0.png','capture1.png','capture2.png']

images = []
for imagePath in imagePaths:
	image = cv.imread(imagePath)
	images.append(image)
print(images[0].shape)

# ret, frame0 = cap0.read()
# images.append(frame0)
# ret, frame1 = cap1.read()
# images.append(frame1)
# ret, frame2 = cap2.read()
# images.append(frame2)

# print(images[0].shape)


stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
	# write the output stitched image to disk
	cv.imwrite('stitchedImage.png', stitched)




# cv.imwrite('capture0.png', frame)
# cap0.release()

# cv.imwrite('capture1.png', frame)
# cap1.release()

# ret, frame = cap2.read()
# cv.imwrite('capture2.png', frame)
# cap2.release()



# index = 0
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#         #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'):
#         break

#     if cv.waitKey(1) == ord('c'):
#         cv.imwrite('testImage' + str(index) + '.png', frame)
#         print('Image saved')
#         index = index + 1
# # When everything done, release the capture
# # cv.destroyAllWindows()