import cv2
from threading import Thread
import time
import numpy as np

class CameraStream:
    def __init__(self, src = 0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.stream.read()
            if self.stopped:
                self.stream.release()
                return

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True