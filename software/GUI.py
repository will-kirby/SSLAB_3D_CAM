import numpy as np
import cv2

from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication
import pyqtgraph as pg
from PyQt5.QtCore import QThread


import time
import matplotlib.pyplot as plt


# async thread
class MovieThread(QThread):
  def __init__(self, camera, window):
    super().__init__()
    self.camera = camera
    self.window = window

  def acquire_movie(self, num_frames):
    movie = []
    for _ in range(num_frames):
      _, frame = self.camera.read()
      movie.append(frame.T)
    return movie

  def run(self):
    print('starting batch')
    frames = self.acquire_movie(200)
    for frame in frames:
      QThread.msleep(500)
      self.window.image_view.setImage(frame)
    print('batch finished')
        


class StartWindow(QMainWindow):
  def __init__(self, camera=None):
    super().__init__()
    self.camera = camera
    self.setWindowTitle("Stitching GUI")

    # register UI elements
    self.central_widget = QWidget()
    self.button_frame = QPushButton('Acquire Frame', self.central_widget)
    self.button_movie = QPushButton('Batch Video', self.central_widget)
    self.layout = QVBoxLayout(self.central_widget)
    self.layout.addWidget(self.button_frame)
    self.layout.addWidget(self.button_movie)
    self.setCentralWidget(self.central_widget)

    self.button_frame.clicked.connect(self.update_image)

    self.image_view = pg.ImageView()
    
    #### FIXME: Change colormap
    colors = [
      (0, 0, 0),
      (4, 5, 61),
      (84, 42, 55),
      (15, 87, 60),
      (208, 17, 141),
      (255, 255, 255)
    ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    self.image_view.setColorMap(cmap)
    self.layout.addWidget(self.image_view)

    #movie stuff
    self.button_movie.clicked.connect(self.start_movie)

  def start_movie(self):
    self.movie_thread = MovieThread(self.camera, self)
    self.movie_thread.start()

  def update_image(self):
    _, frame = self.camera.read()
    # print(frame.shape)
    self.image_view.setImage(frame.T)


if __name__ == '__main__':
  camera = cv2.VideoCapture(0)

  app = QApplication([])
  start_window = StartWindow(camera)
  start_window.show()
  app.exit(app.exec_())