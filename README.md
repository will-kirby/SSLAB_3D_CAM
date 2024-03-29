# SSLAB_3D_CAM

Created in collaboration with [SmartSystems Lab](https://smartsystems.ece.ufl.edu/)

GitHub Link: https://github.com/will-kirby/SSLAB_3D_CAM

- Software: See requirements.txt for python dependencies
- Hardware:
  - Nvidia Jetson Nano
  - 6 USB cameras
    - connected to the Jetson through a USB hub.
  - 1 LEDs connected to GPIO pins on the Jetson
    - used to display the internal state.

## Overview

The system is a 360° Camera with a real-time video feed and the option to write video to a file. The system takes images from each of the 6 cameras, performs a stitching operation to merge the images into one panoramic image, and then outputs that image either to a real-time display or to a file. The system contains an NVIDIA Jetson to process the images, 6 USB cameras to capture images, a USB hub to connect the cameras to the Jetson, and a simple LED circuit to display the system state (LED’s for successful stitch as well as any errors).

The system features a logging tool for testing and debugging purposes. Given arguments, the user can perform variable tests on different camera ranges, recording durations, and stitcher algorithms. Using logs from the testing, we can create a performance report on stitching efficiency.

For a more technical background see [Final Report](Milestones/Final_Report.pdf)

### Behavior Diagram

![Behavior_Diagram](Milestones/Alpha_Design_Behavior.jpg)

# Design Milestones

Check the milestones folder for prototype documentation
Design Project state spreadsheet in [shared folder](https://drive.google.com/drive/folders/1t5Ism2DB_WJJuRsHaPtqMmN0DKmHnhJA?usp=sharing) below contains individual contributions

---

## Available Scripts

### Developer

[Compiling OpenCV with CUDA](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html)

`python3 -m pip install -r requirements.txt`

Installs the required python packages/dependencies using pip package manager

`cd software/functionality_testing && python3 video_loop.py`

Runs the stitching with 2 cameras, outputing video feed. It also allows for the option to save the stitched image and intermediate images and plots the stitch times and error rate.

`cd software/performance_testing && python3 test_runner.py [..args]`

```
Run testing for given amount of time

optional arguments:
-h, --help show this help message and exit
-c CAMERAS, --cameras CAMERAS
Specify the amount of cameras
-d DURATION, --duration DURATION
Specify the duration in seconds to run the test
-i, --image Flag to take one image
```

_software/performance_testing/log_parser.ipynb_

1. Open with [Jupyter Notebook](https://jupyter.org/)
2. Change log_file with path to \*.csv
3. Plots and displays stitching performance

`cd software && python3 IO_video_loop.py`
Runs video loop and displays statuses to LEDs using GPIO. Transfers file using _/software/common/copy.sh_

### User

`cd software/web/static && python3 flaskTest.py`
Automatically run by Jetson on startup to create Flask web server

`cd software/web/static && python3 -m http.server`
Runs a simple http server on client computer. Able to open flaskTestHtml.html to open GUI.

`cd software && python3 samplestitch_socket.py`
Displays GUI for viewing stitched videos over socket protocol

---

## Production Release Specifications

# Example image

![Cylindrical_Warped](screenshots/Preliminary_cylindrical.png)

### Usability

#### Interface

The GUI has been updated to use a web framework that runs on a client computer. This communicates with a Flask server that is configured to run on the Jetson on startup. The Flask server sets up the cameras and image stitcher and has routes to read stitched video. This allows the web client to communicate through http requests over IP through the serial Micro USB cable connected between the Jetson and the client computer.

The GUI currently displays the live stitched video and has the following controls: 
- Toggle image seem blending
- Recalibrate to recalculate homography matrices
- Basic image navigation (zoom, expand)
- Reset to generic saved matrix (persistent state)

The persistent state (saved homography matrix), is a local file on the Jetson. This can be accessed by connecting to the Jetson file system through SSH, or through the recalibrate API call. As a visual indicator during runtime, there is an LED connected to the Jetson that lights up when the flask server is running.

#### Navigation

The user interacts with the system through a GUI that can be opened on a client computer. The GUI contains labeled buttons that state their purposes for resetting and recalibrating the homography matrix. Each of the options were user tested to ensure that exceptions were handled and the programs would not forcibly quit without notifying the user.

#### Perception

Because the GUI is being updated to use a web framework, it has been possible to improve upon the general aesthetic. This includes integrating Pannellum, which utilizes WebGL to allow the user to pan around the stitched video.

The GUI is extremely straightforward with buttons clickable by mouse that are labeled according to function. Other students in the lab were asked to test the camera system using the GUI and buttons. They reported that the GUI was simple and intuitive for what it accomplished. The buttons briefly highlight when clicked. The reset, recalibrate, and blend seems button results in changing image focus and/or quality. Additionally, an LED indicates whether the stitching program is working properly.

The final cylindrically warped image has a non-ideal amount of image warping that causes the sensory feedback of the image to be clearly off relative to the source images, but otherwise successfully stitches multiple different camera images into a 360-degree spanning image. This is currently being iterated upon with improved camera calibration to remove distortion and better composting techniques to improve the quality of the stitch (See figure 1 below).

#### Responsiveness

The custom stitching algorithm currently experiences some latency as image warping takes up the bulk of our start-up time. Camera recalibration occurs within ~10 seconds. If recalibration fails, the system will fall back on the previous homography matrix. The Jetson has an LED that illuminates when the system is working. After the Flask web server is terminated, the video feed will remain on the last captured frame.

### Build Quality

#### Robustness

There are no noticeable glitches. Edge cases, such as the camera homography matrix failing to match key points results in the system falling back on a previously computed matrix. If no matrix has been previously computed, it falls back on a default saved matrix calculated from specific images. When the system boots up, it runs multiple status checks, which prevents improper configuration of the system and unexpected crashes.

Using a USB hub, cameras at a sufficiently low frame rate and resolution (15 FPS and 320x240), and compressed video streams (MJPG), we can prevent sudden failure to read cameras due to power or bandwidth limitations (mostly relevant for reading from 3 to 6 cameras simultaneously).

#### Consistency

The system acts predictably. Unpredictable results, such as serial communication failing will result in another attempt to communicate. If homography calculation fails, then the system falls back on a default precomputed matrix. Additionally, the cameras have been configured to always be read, stitched, and displayed in the same order that corresponds with how they are placed in the case. If there is a camera error (for example, it can't be read), the system will automatically reset after a few minutes.

#### Aesthetic Rigor

The custom 3D printed hexagonal case from a previous build using a Raspberry Pi has been redesigned to hold the Jetson, USB hub, LED indicator, and power switch. This makes it more presentable to the user and limits the possibility of user error when setting up the hardware.

The GUI is a simple webpage containing the stitched video feed and the [aforementioned controls](#Interface).

#### UI Diagram

![Behavior_Diagram](Milestones/UI-Diagram.jpg)

#### External Interface

Once a user has connected to the Jetson, the system can be viewed through the Flask web server. The user can modify the persistent state (camera homography matrix) by clicking on the recalibrate button and return to the persistent state using reset.

#### Persistent State

The Jetson contains a file that holds the camera's homography matrices for the stitching program. These are used to compute perspective warping to overlap each of the images into one ‘panoramic’ image. This homography matrix file is modified when camera recalibration is called, but a default matrix is always available.

#### Internal System

The main processing step occurs when the Jetson captures images from each of the 6 cameras, uses the stored homography matrix to warp the 6 images together, and then outputs the images to the Flask web server.

Additionally, there is a calibration function that searches for key point matches between the images and computes a homography matrix that maps the coordinates of pixels in each image to the final coordinate system of the stitched image. At present, the images from two non-adjacent cameras that are both adjacent to the same camera are mapped to the plane of the camera in between them. The result of this calibration can be stored on the Jetson and loaded by the stitching program.

## Beta Build Vertical Features

- See [Beta Test Plan](Milestones/Beta_Test_Plan.pdf)

### Beta Build Stitcher Demo

[![Beta Build Stitcher Demo](https://img.youtube.com/vi/eHiNH1yC95o/hqdefault.jpg)](https://youtu.be/eHiNH1yC95o)
