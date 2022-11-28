import cv2
from threading import Thread
import time
import numpy as np

class CameraStream:
    def __init__(self, source=0):
        self.source = source  # Source camera
        self.stream = None    # Stream object from cv2.VideoCapture()
        self.thread = None    # Thread that read frames from stream
        self.frame = None     # Last frame captured in thread
        self.ret = None       # Indicate if frame is read correctly
        self.stopped = False  # Flag to stop thread

    def start(self):
        self.stopped = False
        self.stream = cv2.VideoCapture(self.source)
        self.ret, self.frame = self.stream.read()
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        while True:
            self.ret, self.frame = self.stream.read()
            if self.stopped:
                self.stream.release()
                return

    def read(self):
        return self.ret, self.frame

    def stop(self):
        if self.thread is not None:
            self.stopped = True
            self.thread.join()