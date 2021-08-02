import sys
from datetime import datetime
from enum import Enum
from typing import Any
import cv2
import numpy as np
import pandas as pd
import pathlib
import schedule
import time
import threading

ProcessingData = Enum("ProcessingData", "HEAD_POS_ROLL_PITCH_YAW FACE_LANDMARKS LEFT_EYE RIGHT_EYE LEFT_EYE_CENTER "
                                        "RIGHT_EYE_CENTER LEFT_EYE_WIDTH RIGHT_EYE_WIDTH LEFT_EYE_HEIGHT "
                                        "RIGHT_EYE_HEIGHT LEFT_PUPIL_POS RIGHT_PUPIL_POS LEFT_PUPIL_DIAMETER "
                                        "RIGHT_PUPIL_DIAMETER")


def run_continuously(interval=1):
    """Continuously run, while executing pending jobs at each
    elapsed time interval.
    @return cease_continuous_run: threading. Event which can
    be set to cease continuous run. Please note that it is
    *intended behavior that run_continuously() does not run
    missed jobs*. For example, if you've registered a job that
    should run every minute and you set a continuous run
    interval of one hour then your job won't be run 60 times
    at each interval but only once.

    Function taken from https://schedule.readthedocs.io/en/stable/background-execution.html
    """
    cease_continuous_run = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run.is_set():
                schedule.run_pending()
                time.sleep(interval)

    continuous_thread = ScheduleThread()
    continuous_thread.start()
    return cease_continuous_run


# noinspection PyAttributeOutsideInit
class ProcessingLogger:
    def __init__(self, log_folder="processing_data", default_log_file="processing_log.csv"):
        self.__log_folder = log_folder
        self.__log_file = default_log_file

        self.__folder_path = pathlib.Path(__file__).parent / self.__log_folder
        self.__log_file_path = self.__folder_path / self.__log_file

        self.__log_tag = "processing_logger"
        self.__log_data = self.__init_log()
        self.__processed_data = []
        self.__image_data = []
        self.__start_scheduling()

    def __init_log(self):
        # create log folder if it doesn't exist yet
        if not self.__folder_path.is_dir():
            self.__folder_path.mkdir()

        # if the log file already exists and is not empty, read the current log data else create an empty dataframe
        if self.__log_file_path.is_file() and self.__log_file_path.stat().st_size > 0:
            try:
                current_log = pd.read_csv(self.__log_file_path, sep=";")
            except Exception as e:
                sys.stderr.write(f"Error when trying to read file:{e}")
                current_log = pd.DataFrame()
        else:
            current_log = pd.DataFrame()
        return current_log

    def __start_scheduling(self):
        """
        Starts a background thread that logs the currently saved data (a list of dicts) to the csv file;
        this is far more efficient than appending the log data row by row as this actually re-creates the
        whole pandas dataframe completely every time;
        see https://stackoverflow.com/questions/10715965/create-pandas-dataframe-by-appending-one-row-at-a-time

        Another benefit is that it moves the I/O operation (writing to file) happen less often and off the
        main thread.
        """
        # schedule logging to csv file periodically every 5 seconds
        schedule.every(5).seconds.do(self.__save_tracking_data).tag(self.__log_tag)
        # Start the background thread
        self.__logging_job = run_continuously()

    def stop_scheduling(self):
        # cancel current scheduling job
        active_jobs = schedule.get_jobs(self.__log_tag)
        if len(active_jobs) > 0:
            schedule.cancel_job(active_jobs[0])
        # Stop the background thread on the next schedule interval
        self.__logging_job.set()

    def log_frame_data(self, frame_id: float, data: dict[ProcessingData, Any]):
        # ** unpacks the dictionary as key-value pairs
        self.__processed_data.append({'date': datetime.now(), 'frame_id': frame_id, **data})

    def log_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        self.__image_data.append((dirname, filename, image, timestamp))

    def __save_tracking_data(self):
        tracking_df = pd.DataFrame(self.__processed_data)
        self.__log_data = self.__log_data.append(tracking_df)
        self.__log_data.to_csv(self.__log_file_path, sep=";", index=False)
        self.__processed_data.clear()  # reset tracking data

        for entry in self.__image_data:
            self.__save_image(*entry)
        self.__image_data.clear()

    def __save_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        path = self.__folder_path / dirname
        if not path.is_dir():
            path.mkdir()

        # check if empty first as it crashes if given an empty array (e.g. if face / eyes not fully visible)
        if image.size:
            cv2.imwrite(f'{path / filename}__{timestamp}.png', image)
