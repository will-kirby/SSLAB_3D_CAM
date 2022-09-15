import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import subprocess


class App(QMainWindow):
    def __init__(self):
        #Creates window with just a button
        super().__init__()
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        self.button1 = QPushButton(parent=self, text="Start Recording")
        self.button1.clicked.connect(self.button_clicked_on)
        layout.addWidget(self.button1)

        self.child = None
        
	#maybe take some user input for the following settings?
        self.username = "Alan" #for copying: username on remote computer
        self.file = "test_stitched_vid.avi" #file to copy from jetson
        self.remotedir ="" #directory on remote machine to copy to
        self.remotefilename="" #filename on remote machine
        

    def button_clicked_on(self, proc):
        if self.button1.text() == "Start Recording":
            print("Started, input c")
            sys.argv[0] = 'c'
            self.button1.setText("Stop Recording")
            self.button1.clicked.connect(self.button_clicked_off)
            #import IO_video_loop #causes critical errors due to giving main loop control
            self.child = subprocess.Popen(['python3','IO_video_loop.py',self.username, self.file], stdin=subprocess.PIPE) #runs video loop in separate process

    def button_clicked_off(self):
        if self.button1.text() == "Stop Recording":
            self.child.communicate(b'q') #sends q to IO_video_loop stdin
            print("Processing... will show image when done.")
            sys.argv[0] = 'q'
            self.button1.setText("Start Recording")
            self.button1.clicked.disconnect(self.button_clicked_off)

            # Outputting Image
            self.label = QLabel(self)
            self.pixmap = QPixmap("./test-capture0.png")
            self.label.setPixmap(self.pixmap)
            self.setCentralWidget(self.label)
            self.resize(self.pixmap.width(), self.pixmap.height())
            self.show()

    # handler for the signal aka slot
    def onClick(checked):
        print("1")
        checked  # <- only used if the button is checkeable
        print('clicked')

if __name__ == '__main__':
    app = QApplication([])
    gui = App()
    gui.show()
    sys.exit(app.exec())
