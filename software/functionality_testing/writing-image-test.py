import cv2 as cv
import imutils

cameras = []

num_cameras = 2
for i in reversed(range(num_cameras)):
    camera = cv.VideoCapture(f'/dev/camera{i}')
    if not camera.isOpened():
        print(f'Could not open {i}')
    camera.set(cv.CAP_PROP_FPS, 15)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cameras.append(camera)

stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()


s = 1
while s != 0:
    frames = []
    for i, camera in enumerate(cameras):
        status, frame = camera.read()
        frames.append(frame)

    try:
        s, stitch = stitcher.stitch(frames)
        if s == 0:
            print('successful')
            for i, frame in enumerate(frames):
                cv.imwrite(f"frame{i}.jpg",frame)
            cv.imwrite(f"stitched.jpg",stitch)
        else:
            print('unsuccessful')
    except:
        print()

index = 0
while(True):
    frames = []
    for i, camera in enumerate(cameras):
        status, frame = camera.read()
        frames.append(frame)
        cv.imshow(f"Image{i}",frame)

    try:
        s, stitch = stitcher.stitch(frames)
        if s == 0:
            print('successful')
            cv.imshow('stitch', stitch)
        else:
            print('unsuccessful')
    except:
        print()
    



    if cv.waitKey(1) == ord('q'):
        break

    if cv.waitKey(1) == ord('c'):
        print("Writing image")
        cv.imwrite(f"testImage{index}.jpg",frame)
        index += 1

camera.release()
cv.destroyAllWindows()
