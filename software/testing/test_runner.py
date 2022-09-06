import time 
import argparse
import os
import sys
import csv

import random

# load our packages / code
sys.path.insert(1, '../common/')
# import open_camera

def run_test(duration, log_name, num_cameras=4):
  """
  Runs test for a user specified amount of time
  Input:
  duration: int: duration to run the test in seconds
  log_name: str: filepath to csv for logging
  """
  print(f"Running for {duration} seconds")
  print('Logging to', log_name)

  with open(log_name, "w", newline='') as csv_file:
    fieldnames = ['Time', 'Stitched']
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(fieldnames)

    cameras = [open_camera(i) for i in range(num_cameras)]

    end_time = int(time.time())+duration
    while(int(time.time())<=end_time):

      for camera in cameras:
        ret, frame = camera.read()

      writer.writerow([time.perf_counter(), random.randint(0,1)])
      continue

  print('Program stop')
  return

if __name__ == '__main__' :
  parser = argparse.ArgumentParser(description='Run testing for given amount of time')
  parser.add_argument('-d', '--duration', type=int, default=1,
    help='Specify the duration in seconds to run the test')
  parser.add_argument('-i', '--image', type=int, default=1,
    help='Flag to take one image')

  args = parser.parse_args()
  duration = args.duration

  dirname = os.path.dirname(__file__)
  formatted_time = time.strftime("%Y-%m-%d_%H.%M.%S")
  log_name = os.path.join(dirname, f'logs/{formatted_time}.csv')

  image_dir = os.path.join(dirname, 'images')

  run_test(duration, log_name)