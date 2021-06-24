import configparser as cp
import os
import pathlib
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
from PyQt5.QtCore import pyqtSignal
from plyer import notification
import shutil
from paramiko.ssh_exception import SSHException


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
        self.__upload_callback = upload_callback
        self.__log_folder = default_log_folder
        self.__log_file = default_log_file
        self.__log_file_path = pathlib.Path(self.__log_folder + "/" + self.__log_file)
        self.__images_file_path = pathlib.Path(self.__log_folder + "/images")
        self.__log_tag = "logger"
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

        self.transfer_not_finished = False
        self.num_transferred_images = 0

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
        self.image_save_thread = threading.Thread(target=self.__save_images, daemon=True)
        self.image_save_thread.start()

    def __save_images(self, image_format="jpeg"):
        while self.tracking_active:
            if self.image_queue.qsize() > 0:
                filename, image, timestamp = self.image_queue.get()
                # check if image is empty first as it crashes if given an empty array
                # (e.g. if face / eyes not fully visible)
                if image.size:
                    cv2.imwrite(f'{self.__images_file_path}/{filename}__{timestamp}.{image_format}', image)

            time.sleep(0.033)  # wait for 33 ms as this is the maximum we can get new frames from our webcam

    def start_async_upload(self):
        # connect the custom signal to the callback function to update the gui (updating the gui MUST be done from the
        # main thread and not from a background thread, otherwise it would just randomly crash after some time!!)
        self.signal_update_progress.connect(self.__upload_callback)

        self.upload_thread = threading.Thread(target=self.__start_ftp_transfer, daemon=False)  # must be False!
        # self.upload_thread = threading.Thread(target=self.__start_ftp_transfer_byte_version, daemon=False)
        self.upload_thread.start()

    def __start_ftp_transfer(self):
        while self.tracking_active:
            # we use a formatted string as we have a path object and not a string
            all_images = os.listdir(f"{self.__images_file_path}")
            self.all_images_count = len(all_images)
            images_not_transferred = all_images[self.num_transferred_images:]
            if len(images_not_transferred) > 0:
                self.transfer_not_finished = True

            for image in images_not_transferred:
                self.__upload_image(image)
            self.transfer_not_finished = False

    def __upload_image(self, image):
        try:
            self.sftp.put(localpath=f"{self.__images_file_path}/{image}",
                          remotepath=f"{self.user_dir}/images/{image}")
            self.num_transferred_images += 1
            self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
        except Exception as e:
            sys.stderr.write(f"Exception during image upload occurred: {e}")

    def stop_upload(self):
        # while self.transfer_not_finished:
        #    time.sleep(5)  # wait for 5 s before trying again  # TODO this prevents it from finishing the upload

        self.tracking_active = False

        # use join as this is no daemon thread
        # TODO (but also waits till current upload batch is finished which is not very user-friendly)
        # self.upload_thread.join()

        # close the connection to the sftp server
        self.sftp.close()

        # clear the image queue in a thread safe way
        with self.image_queue.mutex:
            self.image_queue.queue.clear()

        # remove local dir after uploading everything
        shutil.rmtree(self.__log_folder)

    """
    def __start_ftp_transfer_byte_version(self):
        while self.tracking_active:
            all_images_count = self.image_queue.qsize()
            if all_images_count > 0:
                self.transfer_not_finished = True
            else:
                self.transfer_not_finished = False

            self.__upload_byte_image(all_images_count)

    def __upload_byte_image(self, all_images_count):
        filename, image, timestamp = self.image_queue.get()  # FIXME: won't work atm as we need a separate queue for
        # this now!
        # transfer image as bytestream directly without saving it locally first
        byte_stream = encode_image(image, img_format=".jpeg")

        try:
            self.sftp.putfo(byte_stream, f"{self.user_dir}/images/{filename}__{timestamp}")
            self.num_transferred_images += 1
            self.signal_update_progress.emit(self.num_transferred_images, all_images_count)
        except Exception as e:
            sys.stderr.write(f"Exception during byte image upload occurred: {e}")

        # self.image_queue.task_done()
    """
