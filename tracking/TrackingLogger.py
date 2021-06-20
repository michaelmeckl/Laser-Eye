import configparser as cp
import os
import pathlib
import sys
import threading
import time
from concurrent import futures
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Any, Optional
import cv2
import numpy as np
import pandas as pd
import pysftp
import schedule
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from plyer import notification
import shutil
from paramiko.ssh_exception import SSHException
from post_processing.image_utils import encode_image
import asyncio
import asyncssh
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import partial

# logging.basicConfig(
#     level=logging.DEBUG, format="%(asctime)s %(thread)s %(funcName)s %(message)s"
# )

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
class Logger(QtWidgets.QWidget):

    signal_update_progress = pyqtSignal(int, int)
    image_queue = Queue()
    # image_queue = Queue(maxsize=128) # TODO

    def __init__(self, upload_callback, default_log_folder="tracking_data", default_log_file="tracking_log.csv"):
        super(Logger, self).__init__()
        self.__upload_callback = upload_callback
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
            time.sleep(5)  # wait for 5 s before trying again
            self.stop_scheduling()  # TODO quickly reaches MaxRecursionDepth if too often called!

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
        try:
            self.sftp = pysftp.Connection(host=self.__hostname, username=self.__username,
                                          password=self.__password, port=self.__port, cnopts=self.__cnopts)
        except SSHException as e:
            sys.stderr.write(f"Couldn't connect to server! Make sure you have an internet connection and the server "
                             f"is running!\nError: {e}")
            notification.notify(title="Connection Error",
                                message="Couldn't connect to server! Make sure you have an internet connection and "
                                        "the server is running!",
                                timeout=5)
            sys.exit(1)

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
        # connect the custom signal to the callback function to update the gui (updating the gui MUST be done from the
        # main thread and not from a background thread, otherwise it would just randomly crash after some time!!)
        self.signal_update_progress.connect(self.__upload_callback)
        self.upload_thread = threading.Thread(target=self.__start_ftp_transfer, daemon=False)
        self.upload_thread.start()

    def run_upload_task(self, sftp, image):
        return sftp.put(localpaths=f"{self.__images_file_path}/{image}",
                        remotepath=f"{self.user_dir}/images/{image}")

    def log_progress(self, source_path, destination_path, bytes_uploaded, total_bytes):
        print(f"Progress:\n{bytes_uploaded} / {total_bytes} (source: {source_path}, dest: {destination_path}")
        if bytes_uploaded == total_bytes:
            self.cc += 1
        if self.cc == 30:
            print(f"Finished: Took {time.time() - self.start}")

    async def transfer(self):
        self.cc = 0
        self.start = time.time()

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=4)

        async with asyncssh.connect(self.__hostname, port=self.__port, username=self.__username,
                                    password=self.__password, known_hosts=None) as conn:
            async with conn.start_sftp_client() as sftp:
                print('connected')
                all_images = os.listdir(f"{self.__images_file_path}")
                all_images_count = len(all_images)
                print("All images currently: ", all_images_count)
                images_not_transferred = all_images[self.num_transferred_images:]
                print("Number transferred:", self.num_transferred_images)
                print("All images not transferred: ", len(images_not_transferred))
                if len(images_not_transferred) > 0:
                    self.transfer_not_finished = True

                logging.info("1.")
                futures = [
                    loop.run_in_executor(executor, partial(self.run_upload_task, sftp, i))
                    for i in images_not_transferred[:31]
                ]

                logging.info("3.")
                results = await asyncio.gather(*futures)
                logging.info("5.")
                for (i, result) in zip(images_not_transferred[:31], results):
                    logging.info("6. Result: %s, %s", i, result)

                # # tasks = (self.run_client(sftp, img) for img in images_not_transferred[:31])
                # tasks = []
                # for img in images_not_transferred[:31]:
                #     task = asyncio.ensure_future(self.run_upload_task(sftp, img))
                #     tasks.append(task)
                # results = await asyncio.gather(*tasks, return_exceptions=True)
                # for result in results:
                #     print(result)

    def start_async_upload_asyncio_version(self):
        asyncio.run(self.transfer())

    def __start_ftp_transfer_byte_version(self):
        while self.tracking_active:
            all_images_count = self.image_queue.qsize()
            if all_images_count > 0:
                self.transfer_not_finished = True

            filename, image, timestamp = self.image_queue.get()
            # transfer image as bytestream directly without saving it locally first
            # byte_stream = encode_image(image, img_format=".png")  #TODO: Time needed to upload 30 images: 13.402 seconds
            byte_stream = encode_image(image, img_format=".jpeg")  #TODO: Time needed to upload 30 images: 5.180 seconds
            self.sftp.putfo(byte_stream, f"{self.user_dir}/images/{filename}__{timestamp}")

            self.num_transferred_images += 1
            self.signal_update_progress.emit(self.num_transferred_images, all_images_count)

            # TODO self.image_queue.task_done()

        self.transfer_not_finished = False

    def __upload_image(self, image):
        # TODO: Time needed to upload 30 images: 12.471 seconds
        # lock.acquire()
        self.sftp.put(localpath=f"{self.__images_file_path}/{image}", remotepath=f"{self.user_dir}/images/{image}")

        self.num_transferred_images += 1
        self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
        # lock.release()

    def __start_ftp_transfer(self):
        while self.tracking_active:
            # we use a formatted string as we have a path object and not a string
            all_images = os.listdir(f"{self.__images_file_path}")
            self.all_images_count = len(all_images)
            print("All images currently: ", self.all_images_count)
            images_not_transferred = all_images[self.num_transferred_images:]
            print("Number transferred:", self.num_transferred_images)
            print("All images not transferred: ", len(images_not_transferred))
            if len(images_not_transferred) > 0:
                self.transfer_not_finished = True

            # TODO: JPEG mit bytestream machts deutlich schneller: 55 -> 18 s (im vergleich zu png bytestream)

            # threads = []
            # lock = threading.Lock()
            for image in images_not_transferred:
                self.__upload_image(image)
                # new_thread = threading.Thread(target=self.__upload_image, args=(image, lock), daemon=True)
                # new_thread.start()
                # threads.append(new_thread)

            # executor.map(self.__upload_image, images_not_transferred)

            # process = []
            # with ThreadPoolExecutor(max_workers=5) as executor:
            #     for image in images_not_transferred:
            #         process.append(executor.submit(self.__upload_image, image))

            # future_to_url = {executor.submit(self.__upload_image, img): img for img in images_not_transferred}
            # for future in futures.as_completed(future_to_url):
            #     url = future_to_url[future]
            #     try:
            #         data = future.result()
            #     except Exception as exc:
            #         print('%r generated an exception: %s' % (url, exc))
            #     else:
            #         print('%r page is %d bytes' % (url, len(data)))

            self.transfer_not_finished = False

    def log_static_data(self, data: dict[TrackingData, Any]):
        # ** unpacks the given dictionary as key-value pairs
        tracking_df = pd.DataFrame({'date': datetime.now(), **data}, index=[0])
        tracking_df.to_csv(self.__log_file_path, sep=";", index=False)

        # with self.sftp.open(f"{self.user_dir}/{self.__log_file}", 'w+', bufsize=32768) as f:
        self.sftp.put(localpath=f"{self.__log_file_path}", remotepath=f"{self.user_dir}/{self.__log_file}")

    def add_image_to_queue(self, filename: str, image: np.ndarray, timestamp: float):
        """
        # if using maxsize for queue init:
        if not self.image_queue.full():
            self.image_queue.put((filename, image, timestamp))
        else:
            time.sleep(0.1)  # Rest for 100ms, we have a full queue
        """
        self.image_queue.put((filename, image, timestamp))

    def log_image(self, filename: str, image: np.ndarray, timestamp: float):
        self.__image_data.append((filename, image, timestamp))

    def __save_images(self):
        for entry in self.__image_data:
            filename, image, timestamp = entry
            # check if image is empty first as it crashes if given an empty array
            # (e.g. if face / eyes not fully visible)
            if image.size:
                pass
                # FIXME: only for testing, disable new images!
                # cv2.imwrite(f'{self.__images_file_path}/{filename}__{timestamp}.png', image)

        self.__image_data.clear()
