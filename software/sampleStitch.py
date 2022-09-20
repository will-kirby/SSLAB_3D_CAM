from CameraSystemClass import CameraSystem
import cv2 as cv



print("Constructing camera system")
cam = CameraSystem([1,2],compressCameraFeed=False)
print("Calculating homography for 0 and 1")
H, matchesMask = cam.calibrateMatrix()
cam.saveHomographyToFile([H])

dst = cam.homographyStitch(frames[0], frames[1], H[0])
cam.displayFrameMatplotlib(dst)

# Save homo
cam.saveHomographyToFile([H])

# Open saved matrix
H = cam.openHomographyFile()


while(1):
    frames = cam.captureCameraImages()


    # Display the resulting frame
    cv.imshow('raw camera', np.concatenate(frames, axis=1))

    # Stitch
    # cv.imshow('pushed images', pushedImages)

    if cv.waitKey(1) == ord('q'):
        break


# print("Reading camera images")
# frames = cam.readFramesFromFiles(["testImage1.jpg", "testImage0.jpg"],"code_testing/")
# frames = cam.readFramesFromFiles(["testImage0.jpg", "testImage1.jpg"],"code_testing/")

# kp, des = cam._findKPandDesMultiple(frames)
# goodMatches = cam.matchDescriptors(des[0], des[1])
# cam.drawMatches(frames[0], kp[0], frames[1], kp[1], goodMatches)

# H, matchesMask = cam.findHomographyFromMatched(goodMatches, kp[0], kp[1])
# cam.drawMatches(frames[0], kp[0], frames[1], kp[1], goodMatches, matchesMask)


# open saved matrix
# H = cam.openHomographyFile()
# print(H)




# cam.overlapStitch(frames)

# img3 = cv.drawKeypoints(frames[0], kp, None, color=(255,0,0))
# cv.imwrite('image0KP.jpg', img3)
