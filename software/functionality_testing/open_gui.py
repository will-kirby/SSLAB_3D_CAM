import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
import subprocess
import paramiko
import time
import threading


class App(QMainWindow):
    def __init__(self):
        #Creates window with just a button
        super().__init__()
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        self.button1 = QPushButton(parent=self, text="Start Recording")
        self.button1.clicked.connect(self.button_clicked_on)
        self.layout.addWidget(self.button1)

        self.button2 = QPushButton(parent=self, text="Recalibrate")
        self.button2.clicked.connect(self.recalibrate)
        self.layout.addWidget(self.button2)
        self.button2.hide()
        
        self.video = QVideoWidget()
        self.video.resize(320,240)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video)

        self.child = None
        self.enable = False
        
	#maybe take some user input for the following settings?
        self.username = "Alan" #for copying: username on remote computer
        self.file = "test_stitched_vid.avi" #file to copy from jetson
        self.remotedir ="" #directory on remote machine to copy to
        self.remotefilename="" #filename on remote machine
        

    #ssh parameters for paramiko:
        self.user = 'lab'
        self.host = '192.168.55.1'
        self.port = 22 #ssh port
        self.password = 'ss_pass'
        #self.cmd = 'python3 -u ~/SSLAB_3D_CAM/software/IO_video_loop.py 2>&1' #-u for unbuffered stdout, 2>&1 redirects stderr to stdout
        self.cmd = 'python3 -u ~/SSLAB_3D_CAM/software/sampleStitch_socket.py 2>&1'
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.stdin, self.stdout, self.stderr = (None, None, None)
     
   
        self.thread1 = None



    def button_clicked_on(self, proc):
        if self.button1.text() == "Start Recording":
            print("Started, input c")
            sys.argv[0] = 'c'
            self.button1.setText("Stop Recording")
            self.button1.clicked.connect(self.button_clicked_off)
            self.button2.show()
            #import IO_video_loop #causes critical errors due to giving main loop control
            #self.child = subprocess.Popen(['python3','IO_video_loop.py',self.username, self.file], stdin=subprocess.PIPE) #runs video loop in separate process
            self.client.connect(self.host, self.port, self.user, self.password)
            self.stdin, self.stdout, self.stderr = self.client.exec_command(self.cmd)
            self.thread1=threading.Thread(target=self.read_output)
            self.thread1.start()
            
    def recalibrate(self):
        if(self.enable):
           self.stdin.write('r')
    
    def read_output(self):
         while True:
            line = self.stdout.readline()
            if line:
                output = line.rstrip()
                print(output)
                if output == "Ready!":
                    self.child = subprocess.Popen(['python','functionality_testing\client.py'])
                if output == "Transmitting!":
                    self.enable = True
            if (self.stdout.channel.exit_status_ready()):
                break
            
    def button_clicked_off(self):
        if self.button1.text() == "Stop Recording" and self.enable:
            #self.child.communicate(b'q') #sends q to IO_video_loop stdin
            #print("Processing... will show image when done.")
            
            self.child.kill()
            self.stdin.write('q')
            
            #sys.argv[0] = 'q'
            self.button2.hide()
            self.button1.setText("Start Recording")
            self.button1.clicked.disconnect(self.button_clicked_off)
            
            self.client.close()
            self.thread1.join()
            self.enable = False
            # Outputting Image
            #self.label = QLabel(self)
            #self.pixmap = QPixmap("./test-capture0.png")
            #self.label.setPixmap(self.pixmap)
            #self.setCentralWidget(self.label)
            #self.resize(self.pixmap.width(), self.pixmap.height())
            #self.show()
            
            #Play stitched video
            #self.player.setMedia(QMediaContent(QUrl.fromLocalFile('C:/Users/thefl/test_stitched_vid.avi'))) #/home/lab/SSLAB_3D_CAM/software/test_stitched_vid.avi
            #self.player.setPosition(0)
            #self.video.show()
            #self.player.play()
            #self.player.mediaStatusChanged.connect(self.statusChanged)

    def statusChanged(self, status):
        if status == QMediaPlayer.EndOfMedia:
           self.video.close()

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