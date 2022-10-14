#!/usr/bin/python3

from PyQt5 import QtGui, uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QFileDialog)
import numpy as np
import sys, os

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from picamera2.previews.qt import QGlPicamera2

basedir = os.path.dirname(__file__)

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1640, 1232)}))

app = QApplication([])

def play_stop_button_pressed():
    global recording
    if recording:
        picam2.stop_encoder()
        window.play_stop_button.setText("Start recording")
        recording = False
    else:
        path, extension = QFileDialog.getSaveFileName(window, 'Save File', "", ".h264")
        if path != '':
            encoder = H264Encoder(10000000)
            encoder.output = FileOutput(path+extension)
            picam2.start_encoder(encoder)
            window.play_stop_button.setText("Stop recording")
            recording = True

window = QWidget()
uic.loadUi(os.path.join(basedir, 'resource/camera_controls.ui'), window)
window.setWindowTitle("Camera App")
window.play_stop_button.clicked.connect(play_stop_button_pressed)
qpicamera2 = QGlPicamera2(picam2, width=640, height=480, keep_ar=True)
window.MainFrame.layout().addWidget(qpicamera2, 0,0,1,3)
recording = False
    

picam2.start()
window.show()
sys.exit(app.exec())