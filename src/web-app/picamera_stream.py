from threading import Thread
from picamera2 import Picamera2
import cv2
import numpy as np

class PiCameraStream:
    def __init__(self):
        self.picam2 = None    # PiCamera2
        self.thread = None    # Thread that read frames from camera
        self.frame = None     # Last frame captured in thread
        self.ret = True       # Indicate if frame is read correctly
        self.stopped = False  # Flag to stop thread

    def picamera_init(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"size": (1640, 1232), "format": 'XRGB8888'}))
        
    def start(self):
        self.stopped = False
        self.picamera_init()
        self.picam2.start()
        self.frame = self.picam2.capture_array()
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while True:
            self.frame = self.picam2.capture_array()
            if self.stopped:
                return

    def read(self):
        return self.ret, self.frame

    def stop(self):
        if self.thread is not None:
            self.stopped = True
            self.thread.join()
            self.picam2.stop()
            self.picam2.close()