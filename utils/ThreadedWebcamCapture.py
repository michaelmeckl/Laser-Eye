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
        # self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # use cv2.CAP_DSHOW backend for hd resolution
        # TODO self.set_hd_resolution()

        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            # break loop if getting frame was not successful
            sys.stderr.write("Unknown error while trying to get current frame!")
            return

        self.c = 0
        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def set_hd_resolution(self):
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # setting to a really high resolution makes opencv use the max resolution of the
        # webcam instead of the 640*480 default for some reason... (no other backend required)
        # see https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 7680)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 4320)

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
            self.c += 1
            if not self.grabbed:
                sys.stderr.write("Unknown error while trying to get current frame!")
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        print(f"Frames on webcam thread: {self.c}")

    def get_stream_fps(self):
        return self.stream.get(cv2.CAP_PROP_FPS)

    def get_stream_dimensions(self):
        return self.stream.get(3), self.stream.get(4)
