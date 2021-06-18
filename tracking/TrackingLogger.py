import configparser as cp
import os
import pathlib
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import cv2
import numpy as np
import pandas as pd
import pysftp
import schedule
import shutil

TrackingData = Enum("TrackingData", "SCREEN_WIDTH SCREEN_HEIGHT CAPTURE_WIDTH CAPTURE_HEIGHT CAPTURE_FPS "
                                    "CORE_COUNT CORE_COUNT_PHYSICAL CORE_COUNT_AVAILABLE SYSTEM SYSTEM_VERSION "
                                    "MODEL_NAME PROCESSOR RAM_OVERALL RAM_AVAILABLE RAM_FREE")


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
    def __init__(self, default_log_folder="tracking_data", default_log_file="tracking_log.csv"):
        self.__log_folder = default_log_folder
        self.__log_file = default_log_file
        self.__log_file_path = pathlib.Path(self.__log_folder + "/" + self.__log_file)
        self.__images_file_path = pathlib.Path(self.__log_folder + "/images")

        # we use the timestamp in ms at the init of the tracking system as the user id as we don't have access to the
        # participant_id from the unity application
        # TODO wir könnten auch einfach ein kleines GUI am anfang einbauen, wo er die ID eingeben muss genau wie in Unity
        self.__user_id = get_timestamp()

        credentials = self.__get_credentials()
        if credentials is None:
            sys.stderr.write("Reading sftp server credentials didn't work! Terminating program...")
            sys.exit(1)

        self.__hostname = credentials["sftp_hostname"]
        self.__username = credentials["sftp_username"]
        self.__password = credentials["sftp_password"]
        self.__port = credentials.getint("sftp_port")
        # NOTE: setting hostkeys to None is highly discouraged as we lose any protection
        # against man-in-the-middle attacks; however, it is the easiest solution and for now it should be fine
        # TODO see https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp for correct solution
        self.__cnopts = pysftp.CnOpts()
        self.__cnopts.hostkeys = None

        self.__init_server_connection()

        self.__log_tag = "logger"
        self.__init_log()
        self.__start_scheduling()

    def __get_credentials(self) -> Optional[cp.SectionProxy]:
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

    def __init_log(self):
        # create log and images folder if they don't exist yet
        folder_path = pathlib.Path(self.__log_folder)
        if not folder_path.is_dir():
            folder_path.mkdir()
        if not self.__images_file_path.is_dir():
            self.__images_file_path.mkdir()

        self.__tracking_data = []
        self.__image_data = []

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
        schedule.every(schedule_interval).seconds.do(self.__save_images).tag(self.__log_tag)
        # Start the background thread
        self.__logging_job = run_continuously()

    def stop_scheduling(self):
        while self.transfer_not_finished:
            time.sleep(0.1)  # wait for 100 ms before trying again
            self.stop_scheduling()  # TODO probably very inefficient?

        # close the connection to the sftp server and remove local directory
        self.sftp.close()

        self.tracking_active = False
        # TODO join as this is no daemon thread:
        self.upload_thread.join()

        # TODO remove local dir after uploading everything (the csv as well!)
        # shutil.rmtree(self.__log_folder)

        # cancel current scheduling job
        active_jobs = schedule.get_jobs(self.__log_tag)
        if len(active_jobs) > 0:
            schedule.cancel_job(active_jobs[0])
        # Stop the background thread on the next schedule interval
        self.__logging_job.set()

    def __init_server_connection(self):
        self.sftp = pysftp.Connection(host=self.__hostname, username=self.__username,
                                      password=self.__password, port=self.__port, cnopts=self.__cnopts)
        # create a directory for this user on the sftp server
        self.user_dir = f"/home/{self.__log_folder}__{self.__user_id}"
        if not self.sftp.exists(self.user_dir):
            self.sftp.makedirs(f"{self.user_dir}/images")  # this automatically creates the parent user dir as well
        else:
            print(f"User dir ({self.user_dir}) already exists for some reason! Creating a new one...")
            self.user_dir = f"{self.user_dir}_1"  # append '_1' to the directory name
            self.sftp.makedirs(f"{self.user_dir}/images")

        self.tracking_active = True
        self.transfer_not_finished = False
        self.num_transferred_images = 0

    def start_async_upload(self):
        self.upload_thread = threading.Thread(target=self.__start_ftp_transfer, daemon=False)
        self.upload_thread.start()

    def __start_ftp_transfer(self):
        while self.tracking_active:
            # TODO make sure only new images are uploaded! (via timestamps? or by saving the last image that was
            #  uploaded? or make a diff between local and remote dir (probably too expensive)?)
            # we use a formatted string as we have a path object and not a string
            all_images = os.listdir(f"{self.__images_file_path}")
            all_images_count = len(all_images)
            print("All images currently: ", all_images_count)
            images_not_transferred = all_images[self.num_transferred_images:]
            print("Number transferred:", self.num_transferred_images)
            print("All images not transferred: ", len(images_not_transferred))
            if len(images_not_transferred) > 0:
                self.transfer_not_finished = True

            for image in images_not_transferred:
                self.sftp.put(localpath=f"{self.__images_file_path}/{image}",
                              remotepath=f"{self.user_dir}/images/{image}")
                self.num_transferred_images += 1

            self.transfer_not_finished = False

            # show content of dir on sftp server
            # self.sftp.cwd('/home/images/eyes')
            # print("Eyes dir on sftp server:", self.sftp.listdir())

    def log_static_data(self, data: dict[TrackingData, Any]):
        # ** unpacks the given dictionary as key-value pairs
        tracking_df = pd.DataFrame({'date': datetime.now(), **data}, index=[0])
        tracking_df.to_csv(self.__log_file_path, sep=";", index=False)
        # TODO transfer directly without saving locally first
        self.sftp.put(localpath=f"{self.__log_file_path}", remotepath=f"{self.user_dir}/{self.__log_file}")

    def log_image(self, filename: str, image: np.ndarray, timestamp: float):
        self.__image_data.append((filename, image, timestamp))

    def __save_images(self):
        for entry in self.__image_data:
            filename, image, timestamp = entry
            # check if image is empty first as it crashes if given an empty array
            # (e.g. if face / eyes not fully visible)
            if image.size:
                cv2.imwrite(f'{self.__images_file_path}/{filename}__{timestamp}.png', image)

        self.__image_data.clear()
