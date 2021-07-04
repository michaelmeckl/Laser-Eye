import os
import queue
import threading
import logging
import random
import sys
import time


# TODO send thread event or pyqt signal to set a flag in the worker to wake up while loop else wait in loop?
#  -> idea: simply don't ask queue all the time, but only if required
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal

logging.basicConfig(format="%(message)s", level=logging.INFO)


class WorkerThread(threading.Thread):
    """ A worker thread that ...
        Ask the thread to stop by calling its join() method.

        Based on this post: https://eli.thegreenplace.net/2011/12/27/python-threads-communication-and-stopping/
    """
    def __init__(self, dir_q, result_q):
        super(WorkerThread, self).__init__()
        self.dir_q = dir_q
        self.result_q = result_q
        self.stoprequest = threading.Event()

    def run(self):
        # As long as we weren't asked to stop, try to take new tasks from the
        # queue. The tasks are taken with a blocking 'get', so no CPU
        # cycles are wasted while waiting.
        # Also, 'get' is given a timeout, so stoprequest is always checked,
        # even if there's nothing in the queue.
        while not self.stoprequest.isSet():
            try:
                dirname = self.dir_q.get(True, 0.05)
                # filenames = list(self._files_in_dir(dirname))
                # self.result_q.put((self.name, dirname, filenames))
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stoprequest.set()
        super(WorkerThread, self).join(timeout)


# TODO oder:
# Step 1: Create a worker class
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        """Long-running task."""
        for i in range(5):
            time.sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()


# TODO oder:
class Runnable(QRunnable):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def run(self):
        # Your long-running task goes here ...
        for i in range(5):
            logging.info(f"Working in thread {self.n}, step {i + 1}/5")
            time.sleep(random.randint(700, 2500) / 1000)

