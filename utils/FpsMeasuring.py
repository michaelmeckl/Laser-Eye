import datetime
import time
from functools import wraps


def timeit(method):
    # Decorator to measure execution time of a function; taken from
    # https://rednafi.github.io/digressions/python/2020/04/21/python-concurrent-futures.html
    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} => {(end_time-start_time)*1000} ms")
        return result
    return wrapper


class FpsMeasurer:
    """
    Class for measuring fps; taken from
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    and slightly adjusted.
    """

    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
        return self._numFrames

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

    def get_current_fps(self):
        elapsed_time = (datetime.datetime.now() - self._start).total_seconds()
        return self._numFrames / elapsed_time if elapsed_time != 0 else self._numFrames

    def show_optimal_fps(self, max_fps):
        fps_ideal = int(1000 / max_fps)
        print(f'* Capture FPS: {max_fps}; ideal wait time between frames: {fps_ideal} ms')
