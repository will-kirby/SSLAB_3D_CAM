# SSLAB_3D_CAM

Code for stitching images for 360 degree camera
Meant for running on Nvidia Jetson with OpenCV CUDA library

# Design Milestones

Check the milestones folder for prototype documentation
Design Project state spreadsheet in [shared folder](https://drive.google.com/drive/folders/1t5Ism2DB_WJJuRsHaPtqMmN0DKmHnhJA?usp=sharing) below contains individual contributions

## Available Scripts

### Developer

`python3 -m pip install -r requirements.txt`

Installs the required python packages/dependencies using pip package manager

`cd software/functionality_testing && python3 video_loop.py`

Runs the stitching with 2 cameras, outputing video feed. It also allows for the option to save the stitched image and intermediate images and plots the stitch times and error rate.

_software/performance_testing/test_runner_

```Run testing for given amount of time

optional arguments:
-h, --help show this help message and exit
-c CAMERAS, --cameras CAMERAS
Specify the amount of cameras
-d DURATION, --duration DURATION
Specify the duration in seconds to run the test
-i, --image Flag to take one image
```

_software/performance_testing/log_parser.ipynb_

```Open with [Jupyter Notebook](https://jupyter.org/)
Change log_file with path to *.csv
Plots and displays stitching performance
```

### User

`cd software && python3 IO_video_loop.py`
Runs video\*loop and displays statuses to LEDs using GPIO. Transfers file using _/software/common/copy.sh_

`cd software && python3 GUI.py`
Displays GUI for viewing stitched images (will be updated to support videos)
