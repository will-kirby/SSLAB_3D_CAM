import cv2 as cv

camera = cv.VideoCapture(0)

index = 0
while(True):
    status, image = camera.read()
    cv.imshow("Image",image)

    if cv.waitKey(1) == ord('c'):
        print("Writing image")
        cv.imwrite(f"testImage{index}.jpg",image)
        index += 1

    if cv.waitKey(1) == ord('q'):
        break


camera.release()
cv.destroyAllWindows()