import cv2 as cv
import numpy as np

# camera = cv.VideoCapture(1)
# _, image = camera.read()
# cv.imwrite("testImage2.jpg",image)

def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # return the visualization
    return vis



img = cv.imread('testImage0.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img2 = cv.imread('testImage1.jpg')
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp_, des = sift.detectAndCompute(gray,None)
kps = np.float32([kp.pt for kp in kp_])
kp_, des2 = sift.detectAndCompute(gray2,None)
kps2 = np.float32([kp.pt for kp in kp_])

matcher = cv.DescriptorMatcher_create("BruteForce")

# returns a list of tuples, each tuple has n (2 in this case) DescripMatchObjects
rawMatches = matcher.knnMatch(des, des2, 2)
print(len(rawMatches), len(kps), len(kps2))

ratio = .7
matches = []
for m in rawMatches:
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        # print(m[0].distance)
        matches.append((m[0].trainIdx, m[0].queryIdx))

print(len(matches))

# computing a homography requires at least 4 matches
if len(matches) > 4:
    # construct the two sets of points
    ptsA = np.float32([kps[i] for (_, i) in matches])
    ptsB = np.float32([kps2[i] for (i, _) in matches])
    # compute the homography between the two sets of points
    (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC,
        4.0)
    print(H)

result = cv.warpPerspective(img, H,
			(img.shape[1] + img2.shape[1], img.shape[0]))
cv.imshow("ResultInit", result)
result[0:img2.shape[0], 0:img2.shape[1]] = img2

cv.imshow("Result", result)
cv.waitKey(0)
cv.destroyAllWindows()

        # print(m[0].distance, m[1].distance)
# print(kps, kps.shape,des.shape)

# img=cv.drawKeypoints(gray,kp,img)
# cv.imwrite('testImage2_keypoints.jpg',img)