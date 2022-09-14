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
def run_test(duration, log_name, num_cameras, algo): # Small change since all parameters can be positional
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
    writer.writerow([f"Duration: {duration}", f"Cameras: {num_cameras}", f"Stitching Algorithm: {algo}"])

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
  # Adding the formatter class will inform users of default values when using -help
  parser = argparse.ArgumentParser(description='Stitching functionality test with variable camera range and timing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-c', '--cameras', type=int, default=4, 
    help='Specify the amount of cameras')
  parser.add_argument('-d', '--duration', type=int, default=10,
    help='Specify the duration in seconds to run the test')
  parser.add_argument('-s', '--stitcher', type=int, default=0, # Stitcher argument. We're using another stitcher for comparison right? I don't know what to put for the 2nd option so I left it as 'other'
    help='Specify stitcher mode. 0 for OpenCV, 1 for *OTHER*')
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
  log_name = os.path.join(dirname, f'logs/{name}.csv')
  image_dir = os.path.join(dirname, 'images')

  # Will have to a similar if statement when selecting stitcher once stitching is integrated.
  if (args.stitcher):
    algo = 'OpenCV Stitcher'
  else:
    algo = 'OTHER'

  run_test(duration, log_name, algo, args.cameras)