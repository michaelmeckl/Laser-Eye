"""
Alternative version for zipping and uploading.
"""

import configparser as cp
import math
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Any, Optional
import cv2
import numpy as np
import pandas as pd
import pysftp
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from plyer import notification
import shutil
import py7zr
from paramiko.ssh_exception import SSHException

from tracking.FpsMeasuring import timeit

TrackingData = Enum("TrackingData", "SCREEN_WIDTH SCREEN_HEIGHT CAPTURE_WIDTH CAPTURE_HEIGHT CAPTURE_FPS "
                                    "CORE_COUNT CORE_COUNT_PHYSICAL CORE_COUNT_AVAILABLE SYSTEM SYSTEM_VERSION "
                                    "MODEL_NAME PROCESSOR RAM_OVERALL RAM_AVAILABLE RAM_FREE")


def get_timestamp() -> float:
    return time.time()


# FIXME: handle internet connection errors by always saving the index of the last transferred image and retrying
#  after 10 seconds from this point again!


# noinspection PyAttributeOutsideInit
class Logger(QtWidgets.QWidget):

    signal_update_progress = pyqtSignal(int, int)
    image_queue = Queue()  # (maxsize=128)

    def __init__(self, upload_callback, default_log_folder="tracking_data", default_log_file="tracking_log.csv"):
        super(Logger, self).__init__()
        self.all_images_count = 0
        self.num_transferred_images = 0
        self.num_transferred_folders = 0
        # FIXME: 1200 or higher
        self.batch_size = 30  # the number of images per subfolder

        self.__upload_callback = upload_callback
        self.__log_folder = default_log_folder
        self.__log_file = default_log_file

        self.__log_folder_path = pathlib.Path(__file__).parent / self.__log_folder  # creates base folder in "tracking/"
        self.__log_file_path = self.__log_folder_path / self.__log_file  # tracking/tracking_data/file.csv
        self.__images_path = self.__log_folder_path / "images"  # tracking/tracking_data/images/
        self.__images_zipped_path = self.__log_folder_path / "images_zipped"  # tracking/tracking_data/images_zipped/
        self.__init_log()

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
        """
        Creates all log and images folders that are needed later.
        """
        if self.__log_folder_path.is_dir():
            # remove old log folder if there is already one (probably some unwanted leftover)
            shutil.rmtree(self.__log_folder_path)

        self.__log_folder_path.mkdir()
        self.__images_path.mkdir()
        self.__images_zipped_path.mkdir()

        self.folder_count = 1
        # create the first subfolder for the first image batch
        self.__get_curr_image_folder().mkdir()

    def __get_curr_image_folder(self):
        return self.__images_path / str(self.folder_count)

    def __init_server_connection(self):
        try:
            self.sftp = pysftp.Connection(host=self.__hostname, username=self.__username,
                                          password=self.__password, port=self.__port, cnopts=self.__cnopts)
        except SSHException as e:
            sys.stderr.write(f"Couldn't connect to server! Make sure you have an internet connection and the server "
                             f"is running, then start the program again!\nError: {e}")
            notification.notify(title="Connection Error",
                                message="Couldn't connect to server! Make sure you have an internet connection and "
                                        "the server is running, then start the program again!",
                                timeout=5)
            sys.exit(1)

        # we use the timestamp in ms at the init of the tracking system as the user id as we don't have access to the
        # participant_id from the unity application
        # TODO wir könnten auch einfach ein kleines GUI am anfang einbauen, wo er die ID eingeben muss wie in Unity
        self.__user_id = get_timestamp()

        # create a directory for this user on the sftp server
        self.user_dir = f"/home/{self.__log_folder}__{self.__user_id}"
        if not self.sftp.exists(self.user_dir):
            self.sftp.makedirs(f"{self.user_dir}/images")  # this automatically creates the parent user dir as well
        else:
            print(f"User dir ({self.user_dir}) already exists for some reason! Creating a new one...")
            self.user_dir = f"{self.user_dir}_1"  # append '_1' to the directory name
            self.sftp.makedirs(f"{self.user_dir}/images")

    def log_csv_data(self, data: dict[TrackingData, Any]):
        # ** unpacks the given dictionary as key-value pairs
        tracking_df = pd.DataFrame({'date': datetime.now(), **data}, index=[0])
        tracking_df.to_csv(self.__log_file_path, sep=";", index=False)

        try:
            self.sftp.put(localpath=f"{self.__log_file_path}", remotepath=f"{self.user_dir}/{self.__log_file}")
        except Exception as e:
            sys.stderr.write(f"Exception during csv upload occurred: {e}")

    def add_image_to_queue(self, filename: str, image: np.ndarray, timestamp: float):
        """
        # if using maxsize for queue:
        if not self.image_queue.full():
            self.image_queue.put((filename, image, timestamp))
        else:
            time.sleep(0.1)  # Rest for 100ms, we have a full queue
        """
        self.image_queue.put((filename, image, timestamp))

    def start_saving_images_to_disk(self):
        self.tracking_active = True
        self.image_save_thread = threading.Thread(target=self.__save_images, name="SaveToDisk", daemon=True)
        self.image_save_thread.start()

    def __save_images(self, img_format="png"):  # jpeg
        while self.tracking_active:
            if self.image_queue.qsize() > 0:
                filename, image, timestamp = self.image_queue.get()
                # check if image is empty first as it crashes if given an empty array
                # (e.g. if face / eyes not fully visible)
                if image.size:
                    image_id = f"{filename}__{timestamp}.{img_format}"
                    image_path = f"{self.__get_curr_image_folder() / image_id}"
                    cv2.imwrite(image_path, image)
                    self.all_images_count += 1
                    self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)

                    # check if the current number of saved images is a multiple of the batch size
                    if (self.all_images_count % self.batch_size) == 0:
                        # a batch of images is finished so we put this one in a queue to be zipped and uploaded
                        self.__start_upload_thread(str(self.folder_count))

                        """
                        zip_file_name = self.__zip_folder(str(self.folder_count))
                        # upload this folder to the server
                        self.__upload_zipped_images(zip_file_name)

                        # update progressbar in gui
                        self.num_transferred_folders += 1
                        self.num_transferred_images += self.batch_size
                        self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
                        """

                        # and create a new folder for the next batch
                        self.folder_count += 1
                        # self.signal_update_progress.emit(self.num_transferred_folders, self.folder_count)
                        self.__get_curr_image_folder().mkdir()

            time.sleep(0.030)  # wait for 30 ms as this is the maximum we can get new frames from our webcam

    def __start_upload_thread(self, folder_name: str):
        with ThreadPoolExecutor() as executor:
            executor.submit(self.__zip_and_upload, folder_name)
        # https://stackoverflow.com/questions/37116721/typeerror-in-threading-function-takes-x-positional-argument-but-y-were-given/37116824#37116824
        # self.zip_and_upload = threading.Thread(target=self.__zip_and_upload, args=(folder_name,),
        #                                        name="ZipUploadThread", daemon=False)
        # self.zip_and_upload.start()

    def __zip_and_upload(self, folder_name: str):
        zip_file_name = self.__zip_folder(folder_name)
        # upload this folder to the server
        self.__upload_zipped_images(zip_file_name)

        # update progressbar in gui
        self.num_transferred_folders += 1
        self.num_transferred_images += self.batch_size
        self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
        # self.signal_update_progress.emit(self.num_transferred_folders, self.folder_count)

    def start_async_upload(self):
        # connect the custom signal to the callback function to update the gui (updating the gui MUST be done from the
        # main thread and not from a background thread, otherwise it would just randomly crash after some time!!)
        self.signal_update_progress.connect(self.__upload_callback)

    @timeit
    def __zip_folder(self, folder_name: str):
        """
        Converts the given folder to a 7zip archive file to speed up the upload.
        """
        folder_path = f"{self.__images_path / folder_name}"
        zip_file_name = folder_name + ".7z"
        zip_filters = None

        with py7zr.SevenZipFile(f"{self.__images_zipped_path / zip_file_name}", 'w', filters=zip_filters) as archive:
            archive.writeall(folder_path)
        return zip_file_name

    @timeit
    def __upload_zipped_images(self, file_name):
        # self.__time_start = get_timestamp()
        try:
            self.sftp.put(localpath=f"{self.__images_zipped_path / file_name}",
                          remotepath=f"{self.user_dir}/images/{file_name}")
                          # callback=self.__on_upload_progress)
        except Exception as e:
            sys.stderr.write(f"Exception during image upload occurred: {e}")

    def __on_upload_progress(self, current, total):
        if current != 0:
            seconds_per_frame = (time.time() - self.__time_start) / current
            eta_seconds = (total - current) * seconds_per_frame
            minutes = math.floor(eta_seconds / 60)
            seconds = round(eta_seconds % 60)
            print(f"Remaining Time: {minutes} min, {seconds} seconds")

    def finish_logging(self):
        # when tracking is stopped the current batch of images will never reach the upload condition, so we do it now
        self.__zip_and_upload(str(self.folder_count))

    def stop_upload(self):
        self.tracking_active = False
        # use join to wait for thread finish as this is no daemon thread (this blocks main ui!)
        # self.upload_thread.join()

        # close the connection to the sftp server
        self.sftp.close()

        # clear the image queue in a thread safe way
        with self.image_queue.mutex:
            self.image_queue.queue.clear()

        # remove local dir after uploading everything
        # TODO enable again:
        # shutil.rmtree(self.__log_folder_path)
