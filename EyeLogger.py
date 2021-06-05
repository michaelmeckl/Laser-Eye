#!/usr/bin/python3
# -*- coding:utf-8 -*-
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


LogData = Enum("LogData", "HEAD_POS_ROLL_PITCH_YAW FACE_LANDMARKS LEFT_EYE RIGHT_EYE LEFT_EYE_CENTER "
                          "RIGHT_EYE_CENTER LEFT_EYE_WIDTH RIGHT_EYE_WIDTH LEFT_EYE_HEIGHT RIGHT_EYE_HEIGHT "
                          "LEFT_PUPIL_POS RIGHT_PUPIL_POS LEFT_PUPIL_DIAMETER RIGHT_PUPIL_DIAMETER")


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
class Logger:
    def __init__(self, log_folder="tracking_data", default_log_file="tracking_log.csv"):
        self.__log_folder = log_folder
        self.__log_file = default_log_file
        self.__log_file_path = pathlib.Path(self.__log_folder + "/" + self.__log_file)

        self.__log_data = self.__init_log()
        self.__tracking_data = []
        self.__start_scheduling()

    def __init_log(self):
        # create log folder if it doesn't exist yet
        folder_path = pathlib.Path(self.__log_folder)
        if not folder_path.is_dir():
            folder_path.mkdir()

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
        schedule_interval = 60  # schedule logging to csv file every minute
        schedule.every(schedule_interval).seconds.do(self.log_tracking_data)
        # Start the background thread
        self.__logging_job = run_continuously()

    def stop_scheduling(self):
        # Stop the background thread on the next schedule interval
        self.__logging_job.set()

    def save_frame(self, frame_id: float, data: dict[LogData, Any]):
        # ** unpacks the dictionary as key-value pairs
        self.__tracking_data.append({'date': datetime.now(), 'frame_id': frame_id, **data})

    def log_tracking_data(self):
        tracking_df = pd.DataFrame(self.__tracking_data)
        self.__log_data = self.__log_data.append(tracking_df)
        self.__log_data.to_csv(self.__log_file_path, sep=";", index=False)
        self.__tracking_data.clear()  # reset tracking data

    def save_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        path = pathlib.Path(self.__log_folder + "/" + dirname)
        if not path.is_dir():
            path.mkdir()

        # check if empty first as it crashes if given an empty array (e.g. if face / eyes not fully visible)
        if image.size:
            cv2.imwrite(f'{path}/{filename}__{timestamp}.png', image)
