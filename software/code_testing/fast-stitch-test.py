# using python 3 and openCV 3.X
# Work in Progress
# See sift-test for current somewhat working thing

import cv2 as cv
import numpy as np

def initCams(numCams, camStartNumber):
    # opens cameras and confirms they are open
    cameras = [cv.VideoCapture(i+camStartNumber) for i in range(numCams)]

    for i, camera in enumerate(cameras):
        if not camera.isOpened():
            raise Exception(f"Cannot open camera {i+camStartNumber}")

    return cameras

def grabImages(cameras):
    frames = [camera.read() for camera in cameras]
    statuses = [frame[0] for frame in frames]
    if 0 in statuses:
        raise Exception("Error grabbing image from one or more frames, exiting")
    images = [frame[1] for frame in frames]

    return images

def calculateHomography(cameras):
    # take image from all cams
    images = grabImages(cameras)
    if not images:
        return None

    # calculate keypoints for each
    sift = cv.SIFT_create()
    gray_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    kp_and_des = [sift.detectAndCompute(g_image, None) for g_image in gray_images]
    
    print(kp_and_des)
        # convert keypoints to array?

    # match keypoints for adjacent cameras, 
        # assume cameras in order are adjacent -> may need to ensure 
        # this somehow, in case restarting the jetson messes up 
        # camera order
    matcher = cv.DescriptorMatcher_create("BruteForce")


    pass

if __name__ == "__main__":
    # initialize cameras
    cameras = initCams(2, 1)

    # initialize homography matrix
    calculateHomography(cameras)

    # while True:
