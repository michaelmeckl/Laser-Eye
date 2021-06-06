"""
Code taken and adapted from
https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/ by Adrian Rosebrock.
"""

import sys
from threading import Thread
import cv2


class WebcamStream:
    def __init__(self, src=0, name="WebcamStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            # break loop if getting frame was not successful
            sys.stderr.write("Unknown error while trying to get current frame!")
            return

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                sys.stderr.write("Unknown error while trying to get current frame!")
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def get_stream_fps(self):
        return self.stream.get(cv2.CAP_PROP_FPS)

    def get_stream_dimensions(self):
        return self.stream.get(3), self.stream.get(4)
