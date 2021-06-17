#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
import os
import shutil
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
import pysftp
import configparser as cp

LogData = Enum("LogData", "HEAD_POS_ROLL_PITCH_YAW FACE_LANDMARKS LEFT_EYE RIGHT_EYE LEFT_EYE_CENTER "
                          "RIGHT_EYE_CENTER LEFT_EYE_WIDTH RIGHT_EYE_WIDTH LEFT_EYE_HEIGHT RIGHT_EYE_HEIGHT "
                          "LEFT_PUPIL_POS RIGHT_PUPIL_POS LEFT_PUPIL_DIAMETER RIGHT_PUPIL_DIAMETER")


def get_timestamp() -> float:
    return time.time()


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

        # we use the timestamp in ms at the init of the tracking system as the user id as we don't have access to the
        # participant_id from the unity application
        # TODO wir könnten auch einfach ein kleines GUI am anfang einbauen, wo er die ID eingeben muss genau wie in Unity
        self.user_id = get_timestamp()

        credentials = self.get_credentials()
        if credentials is None:
            sys.stderr.write("Reading sftp server credentials didn't work! Terminating program...")
            sys.exit(1)

        self.hostname = credentials["sftp_hostname"]
        self.username = credentials["sftp_username"]
        self.password = credentials["sftp_password"]
        self.port = credentials.getint("sftp_port")
        # NOTE: setting hostkeys to None is highly discouraged as we lose any protection
        # against man-in-the-middle attacks; however, it is the easiest solution and for now it should be fine
        # TODO see https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp for correct solution
        self.cnopts = pysftp.CnOpts()
        self.cnopts.hostkeys = None

        self.__log_tag = "logger"
        self.__log_data = self.__init_log()
        self.__tracking_data = []
        self.__image_data = []
        self.__start_scheduling()

        self.init_server_connection()
        # FIXME: start this only once and then let the put run in a while loop in the background until finish
        self.upload_thread = threading.Thread(target=self.start_ftp_transfer, daemon=False)
        self.upload_thread.start()

    def get_credentials(self):
        """
        Read in the credentials for our sftp server.
        Taken from https://gist.github.com/krzysztof-slowinski/59efeef7f9d00b002eed7e0e6636b084
        Credentials file looks like this:
        [dev.sftp]
        sftp_hostname = hostname
        sftp_username = username
        sftp_password = password
        sftp_port = port

        TODO reading in credentials should be replaced later before building exe so we don't have to pass the file to
        every user!
        """
        credentials_file = "sftp_credentials.properties"
        credentials_section = "dev.sftp"

        if os.path.exists(credentials_file):
            credentials = cp.RawConfigParser()
            credentials.read(credentials_file)
            if credentials_section not in credentials:
                print('sftp credentials file is not properly structured!')
                return None
            return credentials[credentials_section]
        else:
            print(f"[Error] Credentials file ({credentials_file}) is not defined!")
            return None

    def init_server_connection(self):
        self.sftp = pysftp.Connection(host=self.hostname, username=self.username,
                                      password=self.password, port=self.port, cnopts=self.cnopts)
        # create a directory for this user on the sftp server
        self.user_dir = f"/home/tracking_data__{self.user_id}"
        if not self.sftp.exists(self.user_dir):
            self.sftp.makedirs(f"{self.user_dir}/eye_regions")
            self.sftp.makedirs(f"{self.user_dir}/eyes")
            self.sftp.makedirs(f"{self.user_dir}/pupils")
        else:
            print(f"User dir ({self.user_dir}) already exists for some reason! Creating a new one...")
            self.user_dir = f"{self.user_dir}_1"  # append '_1' to the directory name
            self.sftp.makedirs(f"{self.user_dir}/eye_regions")
            self.sftp.makedirs(f"{self.user_dir}/eyes")
            self.sftp.makedirs(f"{self.user_dir}/pupils")

    def start_ftp_transfer(self):
        # TODO make sure only new images are uploaded! (via timestamps? or by saving the last image that was
        #  uploaded? or make a diff between local and remote dir (probably too expensive)?)
        for subdir in ["eye_regions", "eyes", "pupils"]:
            # TODO make sure this path exists!
            for image in os.listdir(f"tracking_data/{subdir}"):
                self.sftp.put(localpath=f"tracking_data/{subdir}/{image}", remotepath=f"{self.user_dir}/{subdir}/{image}")

        # self.sftp.cwd('/home/images/eyes')
        # print("Eyes dir on sftp server:", self.sftp.listdir())

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
        schedule_interval = 3  # schedule logging to csv file periodically  # TODO set to 30 later
        schedule.every(schedule_interval).seconds.do(self.__save_tracking_data).tag(self.__log_tag)
        # Start the background thread
        self.__logging_job = run_continuously()

    def stop_scheduling(self):
        self.sftp.put(localpath=f"tracking_data/{self.__log_file}", remotepath=f"{self.user_dir}/{self.__log_file}")

        # close the connection to the sftp server and remove local directory
        self.sftp.close()

        # TODO remove local dir after uploading everything (the csv as well!)
        # shutil.rmtree('tracking_data/')

        # TODO upload log file once at the end as well!

        # cancel current scheduling job
        active_jobs = schedule.get_jobs(self.__log_tag)
        if len(active_jobs) > 0:
            schedule.cancel_job(active_jobs[0])
        # Stop the background thread on the next schedule interval
        self.__logging_job.set()

    def log_frame_data(self, frame_id: float, data: dict[LogData, Any]):
        # ** unpacks the dictionary as key-value pairs
        self.__tracking_data.append({'date': datetime.now(), 'frame_id': frame_id, **data})

    def log_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        self.__image_data.append((dirname, filename, image, timestamp))

    def __save_tracking_data(self):
        tracking_df = pd.DataFrame(self.__tracking_data)
        self.__log_data = self.__log_data.append(tracking_df)
        self.__log_data.to_csv(self.__log_file_path, sep=";", index=False)
        self.__tracking_data.clear()  # reset tracking data

        for entry in self.__image_data:
            self.__save_image(*entry)
        self.__image_data.clear()

    def __save_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        path = pathlib.Path(self.__log_folder + "/" + dirname)
        if not path.is_dir():
            path.mkdir()

        # check if empty first as it crashes if given an empty array (e.g. if face / eyes not fully visible)
        if image.size:
            cv2.imwrite(f'{path}/{filename}__{timestamp}.png', image)
