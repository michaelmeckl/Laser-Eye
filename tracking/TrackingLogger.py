import configparser as cp
import os
import pathlib
import shutil
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Any, Optional
import cv2
import numpy as np
import pandas as pd
import pysftp
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QThreadPool
from paramiko.ssh_exception import SSHException
from plyer import notification
from py7zr import FILTER_BROTLI, SevenZipFile


# whitespaces at the end are necessary!!
TrackingData = Enum("TrackingData", "SCREEN_WIDTH SCREEN_HEIGHT CAPTURE_WIDTH CAPTURE_HEIGHT CAPTURE_FPS "
                                    "CORE_COUNT CORE_COUNT_PHYSICAL CORE_COUNT_AVAILABLE CPU_FREQUENCY_MHZ GPU_INFO "
                                    "SYSTEM SYSTEM_VERSION MODEL_NAME PROCESSOR RAM_OVERALL_GB RAM_AVAILABLE_GB "
                                    "RAM_FREE_GB")


def get_timestamp() -> float:
    """
    Returns the current (unix) timestamp in milliseconds.
    """
    return time.time_ns() / 1000000


# FIXME: handle internet connection errors by always saving the index of the last transferred image and retrying
#  after 10 seconds from this point again!


# noinspection PyAttributeOutsideInit
class Logger(QtWidgets.QWidget):

    signal_update_progress = pyqtSignal(int, int)  # used to communicate upload progress with gui on the main thread

    image_queue = Queue()
    upload_queue = Queue()

    def __init__(self, upload_callback, default_log_folder="tracking_data", default_log_file="tracking_log.csv"):
        super(Logger, self).__init__()
        self.all_images_count = 0
        self.num_transferred_images = 0
        self.num_transferred_folders = 0
        self.batch_size = 250  # the number of images per subfolder

        self.__upload_callback = upload_callback
        self.__log_folder = default_log_folder
        self.__log_file = default_log_file

        # get path depending on whether this is started as an exe or normally
        if getattr(sys, 'frozen', False):
            # the program was started as exe
            self.__is_exe = True
            self.__log_folder_path = pathlib.Path(self.__log_folder)
        else:
            self.__is_exe = False
            # creates base folder in "tracking/"
            self.__log_folder_path = pathlib.Path(__file__).parent / self.__log_folder

        self.__log_file_path = self.__log_folder_path / self.__log_file  # tracking/tracking_data/file.csv
        self.__images_path = self.__log_folder_path / "images"  # tracking/tracking_data/images/
        self.__images_zipped_path = self.__log_folder_path / "images_zipped"  # tracking/tracking_data/images_zipped/
        if self.__is_exe:
            # If we are starting this as exe, we also have a folder with the unity logs that need to be uploaded as well
            # The python exe needs to be in the same folder as the game is!
            self.unity_log_folder = self.__log_folder_path.parent / "Game_Data" / "StudyLogs"
            self.uploading_game_data = False

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
            notification.notify(title="Verbindungsfehler",
                                message="Die Verbindung zum Server konnte nicht hergestellt werden! Bitte stellen Sie "
                                        "sicher, dass sie während des Trackings eine stabile Internetverbindung haben!",
                                timeout=5)
            sys.exit(1)

        # we use the timestamp in ms at the init of the tracking system as the user id as we don't have access to the
        # participant_id from the unity application (this should be fine as it is highly unlikely that 2 people start
        # at the exact same millisecond)
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

    def log_too_early_quit(self):
        # log when a user quit too early (i.e. if the image upload hasn't been finished yet)
        try:
            with pysftp.Connection(host=self.__hostname, username=self.__username, password=self.__password,
                                   port=self.__port, cnopts=self.__cnopts) as sftp_connection:
                # open automatically creates the file on the server
                sftp_connection.open(remote_file=f"{self.user_dir}/user_quit_too_early.txt", mode="w+")
        except Exception as e:
            sys.stderr.write(f"Exception during quit too early upload occurred: {e}")

    def upload_game_data(self):
        # only upload game data if this is actually running as an exe, otherwise we don't have game data!
        if not self.__is_exe:
            return
        self.uploading_game_data = True

        # zip the game data folder first
        file_name = "game_log.7z"
        zipped_location = self.__log_folder_path.parent / file_name
        with SevenZipFile(f"{zipped_location}", 'w') as archive:
            archive.writeall(self.unity_log_folder)

        try:
            # we need a new pysftp connection to not get in conflict with the existing image upload on the other thread!
            with pysftp.Connection(host=self.__hostname, username=self.__username, password=self.__password,
                                   port=self.__port, cnopts=self.__cnopts) as sftp_connection:
                # on remote server we always have a POSIX-like path system so there is no need for pathlib in remotepath
                sftp_connection.put(localpath=f"{zipped_location}", remotepath=f"{self.user_dir}/{file_name}")
        except Exception as e:
            sys.stderr.write(f"Exception during game data upload occurred: {e}")

        self.uploading_game_data = False

    def add_image_to_queue(self, filename: str, image: np.ndarray, timestamp: float):
        self.image_queue.put((filename, image, timestamp))

    def start_saving_images_to_disk(self):
        self.tracking_active = True
        self.image_save_thread = threading.Thread(target=self.__save_images, name="SaveToDisk", daemon=True)
        self.image_save_thread.start()

    def __save_images(self, img_format="png"):
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
                    # self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)

                    # check if the current number of saved images is a multiple of the batch size
                    if (self.all_images_count % self.batch_size) == 0:
                        # a batch of images is finished so we put this one in a queue to be zipped and uploaded
                        self.upload_queue.put(str(self.folder_count))

                        # and create a new folder for the next batch
                        self.folder_count += 1
                        self.signal_update_progress.emit(self.num_transferred_folders, self.folder_count)
                        self.__get_curr_image_folder().mkdir()

            # wait to prevent constantly asking the queue for new images which would have a huge impact on the fps
            # TODO: this can actually cause huge memory problems if waiting too long as the queue will expand rapidly!
            time.sleep(0.01)

    """
    one thread (100 batch size):
        Time needed to upload 100 images: 9.868 seconds
        Time needed to upload 500 images: 28.478 seconds
        Time needed to upload 1200 images: 62.630 seconds
    
    with thread pool (100 batch size):
        Time needed to upload 100 images: 10.751 seconds
        Time needed to upload 500 images: 24.482 seconds
        Time needed to upload 1200 images: 47.043 seconds
    
    (both with ~12 mbit upload speed)
    """
    def start_async_upload(self):
        # connect the custom signal to the callback function to update the gui (updating the gui MUST be done from the
        # main thread and not from a background thread, otherwise it would just randomly crash after some time!!)
        self.signal_update_progress.connect(self.__upload_callback)

        threadCount = QThreadPool.globalInstance().maxThreadCount()  # get the maximal thread count that we can use
        # create a lock for the uploading process (only one thread at a time can actually transfer the data via sftp)
        self.upload_lock = threading.Lock()
        # self.upload_pool = []
        for i in range(threadCount):
            # must not be a daemon thread here!
            upload_thread = threading.Thread(target=self.__start_ftp_transfer, name="UploadThread", daemon=False)
            upload_thread.start()
            # self.upload_pool.append(upload_thread)

    def __start_ftp_transfer(self):
        while self.tracking_active:
            if self.upload_queue.qsize() > 0:
                file_name = self.upload_queue.get()

                # zip the the folder with the given name
                zip_file_name = self.__zip_folder(file_name)
                # upload this folder to the server
                self.upload_lock.acquire()
                self.__upload_zipped_images(zip_file_name)
                self.upload_lock.release()

                # update progressbar in gui
                self.num_transferred_folders += 1
                self.num_transferred_images += self.batch_size
                # self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
                self.signal_update_progress.emit(self.num_transferred_folders, self.folder_count)

            time.sleep(0.01)  # wait for the same amount of time as the other queue

    def __zip_folder(self, folder_name: str):
        """
        Converts the given folder to a 7zip archive file to speed up the upload.
        """
        folder_path = f"{self.__images_path / folder_name}"
        zip_file_name = folder_name + ".7z"
        zip_filters = [{'id': FILTER_BROTLI, 'level': 3}]
        # for others compression options, see https://py7zr.readthedocs.io/en/latest/api.html#compression-methods

        with SevenZipFile(f"{self.__images_zipped_path / zip_file_name}", 'w', filters=zip_filters) as archive:
            archive.writeall(folder_path)
        return zip_file_name

    def __upload_zipped_images(self, file_name):
        try:
            self.sftp.put(localpath=f"{self.__images_zipped_path / file_name}",
                          remotepath=f"{self.user_dir}/images/{file_name}")
        except Exception as e:
            sys.stderr.write(f"Exception during image upload occurred: {e}")

    def finish_logging(self):
        # when tracking is stopped the last batch of images will never reach the
        # upload condition (as it won't be a full batch), so we set it manually
        self.upload_queue.put(str(self.folder_count))

    def stop_upload(self):
        self.tracking_active = False

        if self.__is_exe:
            # if uploading the game data hasn't finished yet, wait for it first!
            while self.uploading_game_data:
                time.sleep(0.1)  # wait for 100 ms, then try again
                self.stop_upload()

        # for thread in self.upload_pool:
        #    thread.join()

        # close the connection to the sftp server
        self.sftp.close()
        # clear the image and upload queues in a thread safe way
        with self.image_queue.mutex:
            self.image_queue.queue.clear()
        with self.upload_queue.mutex:
            self.upload_queue.queue.clear()

        # remove local dir after uploading everything
        shutil.rmtree(self.__log_folder_path)
        # TODO remove game data folder as well if running as exe??
        # if self.__is_exe:
        #    shutil.rmtree(self.__log_folder_path.parent / "Game_Data")
