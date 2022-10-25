import cv2

def open_camera(number, debug=False):
  camera = cv2.VideoCapture(f'/dev/video{number}')
  camera.set(cv2.cv.CAP_PROP_FPS, 15)
  camera.set(cv2.CV_CAP_PROP_FRAME_WIDTH, 320)
  camera.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, 240)
  camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

  if debug:
    print(f'Opening camera {number}')

  if not camera.isOpened():
    print(f"Cannot open camera {number}")
    exit()
  
  return camera