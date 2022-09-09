import time 
import argparse
import os
import sys
import csv

import cv2
import imutils

import random

# load our packages / code
sys.path.insert(1, '../common/')
# import open_camera

# put number of cameras, stitching algo
def run_test(duration, log_name, num_cameras=4, stitching_algo='OpenCV Stitcher'):
  """
  Runs test for a user specified amount of time
  Input:
  duration: int: duration to run the test in seconds
  log_name: str: filepath to csv for logging
  """
  print(f"Running for {duration} seconds")
  print('Logging to', log_name)

  with open(log_name, "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([f"Duration: {duration}", f"Cameras: {num_cameras}", f"Stitching Algorithm: {stitching_algo}"])

    fieldnames = ['Start Time', 'End Time', 'Status']
    writer.writerow(fieldnames)

    # cameras = [open_camera(i) for i in range(num_cameras)]
    # stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()

    end_time = int(time.time())+duration
    while(int(time.time())<=end_time):

      # frames = []
      # for camera in cameras:
      #   ret, frame = camera.read()
      #   if not ret:
      #     print(f"Can't receive frame (stream end?). Exiting ...")
      #     break

      #   frames.append(frame)

      # (status, stitched) = stitcher.stitch(frames)
      start = time.perf_counter()
      time.sleep(.1)
      end = time.perf_counter()

      status = random.randint(0,1)
      writer.writerow([start, end, status])

  print('Program stop')
  return

if __name__ == '__main__' :
  parser = argparse.ArgumentParser(description='Run testing for given amount of time')
  parser.add_argument('-c', '--cameras', type=int, default=4,
    help='Specify the amount of cameras')
  parser.add_argument('-d', '--duration', type=int, default=10,
    help='Specify the duration in seconds to run the test')
  parser.add_argument('-i', '--image', action='store_true', default=False,
    help='Flag to take one image')

  args = parser.parse_args()
  duration = args.duration

  dirname = os.path.dirname(__file__)
  formatted_time = time.strftime("%Y-%m-%d_%H.%M.%S")
  log_name = os.path.join(dirname, f'logs/{formatted_time}.csv')

  image_dir = os.path.join(dirname, 'images')

  run_test(duration, log_name, args.cameras)