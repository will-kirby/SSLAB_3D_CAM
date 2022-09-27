# SSLAB_3D_CAM

Created in collaboration with [SmartSystems Lab](https://smartsystems.ece.ufl.edu/)

GitHub Link: https://github.com/will-kirby/SSLAB_3D_CAM

- Software: See requirements.txt for python dependencies
- Hardware:
  - 6 USB cameras are connected to the Jetson through a USB hub.
  - 2 LEDs connected to GPIO pins on the Jetson
    - used to display the internal state.

## Overview

The system is a 360° Camera with a real-time video feed and the option to write video to a file. The system takes images from each of the 6 cameras, performs a stitching operation to merge the images into one panoramic image, and then outputs that image either to a real-time display or to a file. The system contains an NVIDIA Jetson to process the images, 6 USB cameras to capture images, a USB hub to connect the cameras to the Jetson, and a simple LED circuit to display the system state (LED’s for successful stitch as well as any errors).

The system features a logging tool for testing and debugging purposes. Given arguments, the user can perform variable tests on different camera ranges, recording durations, and stitcher algorithms. Using logs from the testing, we can create a performance report on stitching efficiency.

# Design Milestones

Check the milestones folder for prototype documentation
Design Project state spreadsheet in [shared folder](https://drive.google.com/drive/folders/1t5Ism2DB_WJJuRsHaPtqMmN0DKmHnhJA?usp=sharing) below contains individual contributions

---

## Available Scripts

### Developer

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

### User

`cd software && python3 IO_video_loop.py`
Runs video loop and displays statuses to LEDs using GPIO. Transfers file using _/software/common/copy.sh_

`cd software && python3 GUI.py`
Displays GUI for viewing stitched images (will be updated to support videos)

---

## Beta Build

#### Demo video

[![Beta Build Stitcher Demo](https://img.youtube.com/vi/eHiNH1yC95o/hqdefault.jpg)](https://youtu.be/eHiNH1yC95o)

#### UI Diagram

![UI_Diagram](Milestones/UI-Diagram.jpg)

## Specifications (Alpha Build)

### Usability

#### Interface

The Jetson connects to a standard monitor, mouse, and keyboard. The stitching program can then be started by running a python file in the terminal. An optional GUI can be displayed if desired (with the command line arg), that contains a start and stop button for recording a stitched image. Otherwise, a window will open with a real-time stitched image. Debug and status information is printed to the terminal, and LEDs light up based on program status (red LED if error, green for running).

#### Navigation

The program is started through an Ubuntu terminal, through the Jetson’s operating system, when the Jetson is connected to a monitor and keyboard. A GUI script can be run. The GUI contains clickable start and stops capture buttons. Otherwise, a window with live capture displays, and can be exited by pressing ‘q’ on the window.

#### Perception

The GUI is extremely straightforward with a singular interactable button labeled “Start recording”, that can be clicked on by a mouse cursor. When the camera-operating script is run by a button click, there are LED lights to indicate several pieces of debugging information, like whether or not it is still running and errors encountered, and the button prompt changes to “Stop recording”. After that button has been pressed, the camera-operating script is stopped, and a notice is an output that states that the stitcher is processing the image and will display it when ready. Upon stitch completion, the user is presented with the stitched image

#### Responsiveness

Due to the nature of the project, the main execution loop captures video across multiple cameras. Using the GUI, the user will be able to view the stitched images and control the video capture. When inputs are sent using the GUI, they are immediately passed into the video capturing script, and once read by the secondary script should start or stop a capture. While the image stitching is in progress, the start & stop recording buttons are not accessible due to the script processing the recorded images or video into a deliverable stitch. After the completion of the image stitch, the buttons can be utilized again.

### Build Quality

#### Robustness

The camera system is contained in a plastic 3d printed hexagonal shell, which protects the cameras and other electronic components. The program runs through multiple status checks, such as checking that all the cameras are successfully connected and running, which prevents unexpected crashes. Failure to stitch frames does not crash the program, but sets a status flag and causes an LED to light up.

Using an externally powered USB hub, lowering frame rate, downscaling the resolution, and compressing video streams prevents sudden failure to read cameras due to power or bandwidth limitations (mostly relevant for reading from 3 to 6 cameras simultaneously)

#### Consistency

As stated above, the program runs multiple status checks. This includes checking which cameras are connected correctly, and which cameras successfully captured an image, as well as the outcome of the image stitch. Ensuring that all of these checks are passed helps ensure that no unexpected behavior occurs.

To further improve performance consistency, status checks and timings are logged when running the repeatable test. These logs are parsed to provide performance metrics. These will be used to test/iterate different stitching algorithms/optimizations for future builds.

#### Aesthetic Rigor

For software, the current GUI implementation’s aesthetic is very simplistic. It opens a small window with a “Start recording” prompt. Upon being clicked, that is relayed to the user through the button changing to a “Stop recording” prompt. When the user clicks this button again, a notice that stitching is in progress is output. Upon completion of the stitching, the resulting image is displayed.

For physical artifacts, there is a hardshell plastic casing that protects the cameras & most of the internal electronics. At the moment, the USB hub utilized does not fit in the shell, though this does not affect the performance of the cameras or the security that the shell provides to those parts. This can be remedied by either purchasing a more compact hub in the future or printing a new plastic shell to accommodate these size concerns.

### Vertical Features

- See [Alpha Test Plan](/Milestones/Alpha_Test_Plan.pdf) for User Diagram

#### External Interface

The program contains a GUI, also a window with a real-time display of the stitched image

#### Persistent State.

On the GUI script button press, the primary video_loop script is called. Upon completion, it saves an image or video output. This output is then utilized by the GUI script after it is available to display the output image/video.

#### Internal System

The main processing step of the system is stitching 6 camera images together into one panoramic image. This is done every cycle. For this step, we have 3 different algorithms that can be used, with different performance and quality for each.

The simplest algorithm is shifting the images from each camera until they overlap. This has the benefit of being extremely fast, but the images do not line up perfectly. The second algorithm is to use the OpenCV image stitcher. This has the benefit of near-perfect quality, However, images are frequently dropped from the end product, as keypoint matching fails. It also takes the longest amount of time. A third algorithm is being developed in an attempt at a middle ground. It does keypoint detection and matching once to create a homography matrix, then uses this homography matrix to stitch the images together. This has the benefit of being fast, at the cost of slightly lower quality than the OpenCV stitching algorithm.
