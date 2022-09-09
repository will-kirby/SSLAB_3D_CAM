import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap

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

    def button_clicked_on(self, proc):
        if self.button1.text() == "Start Recording":
            print("Started, input c")
            sys.argv[0] = 'c'
            self.button1.setText("Stop Recording")
            self.button1.clicked.connect(self.button_clicked_off)
            import video_loop

    def button_clicked_off(self):
        if self.button1.text() == "Stop Recording":
            print("Processing... will show image when done.")
            self.button1.setText("Start Recording")
            sys.argv[0] = 'q'
            self.button1.clicked.disconnect(self.button_clicked_off)

            # Outputting Image
            self.label = QLabel(self)
            self.pixmap = QPixmap("./image.jpg")
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
