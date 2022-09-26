import time 
import argparse
import os
import sys
import csv
from CameraSystemClass import CameraSystem
import cv2 as cv
import imutils

import random

# load our packages / code
sys.path.insert(1, '../common/')
# import open_camera

# put number of cameras, stitching algo
def run_test(duration, log_name, algo, num_cameras): # Small change since all parameters can be positional
  """
  Runs test for a user specified amount of time
  Input:
  duration: int: duration to run the test in seconds
  log_name: str: filepath to csv for logging
  """

  with open(log_name, "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([f"Duration: {duration}", f"Cameras: {num_cameras}", f"Stitching Algorithm: {algo}"])

    fieldnames = ['Start Time', 'End Time', 'Status']
    writer.writerow(fieldnames)
    # ///////// OpenCV
    if (algo == 'OpenCV'):
      cameras = []
      for i in range(num_cameras):
        camera = cv.VideoCapture("/dev/camera"+str(i))
        camera.set(cv.CAP_PROP_FPS, 15)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cameras.append(camera)
      for i, camera in enumerate(cameras):
        if not camera.isOpened():
          print(f"Cannot open camera {i}")
          exit()
      stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()

    # ///////// Homography (Based on sampleStitch.py, commit on 9/25)
    elif (algo == 'Homography'):
      #camArray = list(range(0,num_cameras))
      #cam = CameraSystem(camArray,compressCameraFeed=False)
      cam = CameraSystem([0,1,2],compressCameraFeed=False)
      start = time.perf_counter()
      Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
      end = time.perf_counter()
      writer.writerow([start, end, 0, 1])
      cam.saveHomographyToFile([Hl, Hr])
      Hl, Hr = cam.openHomographyFile()

    print('Opened Cameras')
    print(f"Running for {duration} seconds")
    print('Logging to', log_name)

    end_time = int(time.time())+duration

    # ///// Start loop
    while(int(time.time())<=end_time):
      if (algo == 'Homography'):
        frames = cam.captureCameraImages()
        start = time.perf_counter()
        stitched = cam.homographyStitch(frames[0], frames[1], frames[2],  Hl, Hr)
        end = time.perf_counter()
        writer.writerow([start, end, 0])
        #if (duration % 10 == 0):
        #  start = time.perf_counter()
        #  Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        #  end = time.perf_counter()
        #  writer.writerow([start, end, 0, 1])

      if (algo == 'OpenCV'):
        frames = []
        for camera in cameras:
          ret, frame = camera.read()
          if not ret:
            print(f"Can't receive frame (stream end?). Exiting ...")
            break
        start = time.perf_counter()
        (status, stitched) = stitcher.stitch(frames)
        end = time.perf_counter()
        writer.writerow([start, end, status])
        print('log')

  if algo == 'OpenCV':
      camera.release()
  print('Program stop')
  return

if __name__ == '__main__' :
  # Adding the formatter class will inform users of default values when using -help
  parser = argparse.ArgumentParser(description='Stitching functionality test with variable camera range and timing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-c', '--cameras', type=int, default=2, 
    help='Specify the amount of cameras')
  parser.add_argument('-d', '--duration', type=int, default=10,
    help='Specify the duration in seconds to run the test')
  parser.add_argument('-s', '--stitcher', type=int, default=0, # Stitcher argument. We're using another stitcher for comparison right? I don't know what to put for the 2nd option so I left it as 'other'
    help='Specify stitcher mode. 0 for OpenCV, 1 for Homography, 2 for Other') # Shifting for other?
  parser.add_argument('-i', '--image', action='store_true', default=False,
    help='Flag to take one image')
  parser.add_argument('-n', '--name', type=str, default='current time', 
    help='Specify a name for the created log')

  args = parser.parse_args()
  duration = args.duration

  dirname = os.path.dirname(__file__)
  if (args.name == 'current time'):
    name = time.strftime("%Y-%m-%d_%H.%M.%S")
  else:
    name = args.name
  log_name = os.path.join(dirname, f'logs/{name}/{name}.csv')
  image_dir = os.path.join(dirname, f'logs/{name}')

  # Will have to a similar if statement when selecting stitcher once stitching is integrated.
  if (args.stitcher == 0):
    algo = 'OpenCV'
  elif (args.stitcher == 1):
    algo = 'Homography'
  else:
    algo = 'Other'
  num_cameras = args.cameras
  #run_test(duration, log_name, algo, args.cameras)
  with open(log_name, "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([f"Duration: {duration}", f"Cameras: {num_cameras}", f"Stitching Algorithm: {algo}"])

    fieldnames = ['Start Time', 'End Time', 'Status']
    writer.writerow(fieldnames)
    # ///////// OpenCV
    if (algo == 'OpenCV'):
      cameras = []
      for i in range(num_cameras):
        camera = cv.VideoCapture("/dev/camera"+str(i))
        camera.set(cv.CAP_PROP_FPS, 15)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cameras.append(camera)
      for i, camera in enumerate(cameras):
        if not camera.isOpened():
          print(f"Cannot open camera {i}")
          exit()
      stitcher = cv.createStitcher() if imutils.is_cv3() else cv.Stitcher_create()

    # ///////// Homography (Based on sampleStitch.py, commit on 9/25)
    elif (algo == 'Homography'):
      #camArray = list(range(0,num_cameras))
      #cam = CameraSystem(camArray,compressCameraFeed=False)
      cam = CameraSystem([0,1,2],compressCameraFeed=False)
      start = time.perf_counter()
      Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
      end = time.perf_counter()
      writer.writerow([start, end, 0, 1])
      cam.saveHomographyToFile([Hl, Hr])
      Hl, Hr = cam.openHomographyFile()

    print('Opened Cameras')
    print(f"Running for {duration} seconds")
    print('Logging to', log_name)

    frames = []
    for camera in cameras:
      ret, frame = camera.read()
    (status, stitched) = stitcher.stitch(frames)
    cv.imwrite(f'{image_dir}/capture{i}.png', frame)
    cv.imwrite(f'{image_dir}/stitched.png', stitched)
    print(f'capture{i}.png saved')
    exit()

    end_time = int(time.time())+duration

    # ///// Start loop
    while(int(time.time())<=end_time):
      if (algo == 'Homography'):
        frames = cam.captureCameraImages()
        start = time.perf_counter()
        stitched = cam.homographyStitch(frames[0], frames[1], frames[2],  Hl, Hr)
        end = time.perf_counter()
        writer.writerow([start, end, 0])
        #if (duration % 10 == 0):
        #  start = time.perf_counter()
        #  Hl, Hr = cam.calibrateMatrixTriple(frames[0], frames[1], frames[2])
        #  end = time.perf_counter()
        #  writer.writerow([start, end, 0, 1])

      if (algo == 'OpenCV'):
        frames = []
        for camera in cameras:
          ret, frame = camera.read()
          
          if not ret:
            print(f"Can't receive frame (stream end?). Exiting ...")
            break
        start = time.perf_counter()
        (status, stitched) = stitcher.stitch(frames)
        end = time.perf_counter()
        writer.writerow([start, end, status])
        print('log')

  if algo == 'OpenCV':
      camera.release()
  print('Program stop')
