import concurrent
import math
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import cv2
import numpy as np
import pysftp
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication
from plyer import notification
import shutil
import py7zr
from paramiko.ssh_exception import SSHException
from py7zr import FILTER_LZMA, FILTER_DEFLATE, FILTER_BZIP2, FILTER_BROTLI, FILTER_LZMA2, PRESET_DEFAULT, FILTER_ZSTD, \
    FILTER_X86, FILTER_DELTA
from tracking.FpsMeasuring import timeit, FpsMeasurer


def get_timestamp() -> float:
    """
    Returns the current (unix) timestamp in milliseconds.
    """
    return time.time_ns() / 1000000


# Functions below taken from https://www.thepythoncode.com/article/get-directory-size-in-bytes-using-python
def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total


def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"


"""
100:
    798
    1243
    1248
    1248
    1254
    1906
    1933
    905
    881
    807
    1194
    1240
    1239
    1245
    
    upload:
    74608
    74798
    74812
    75695
    21323

500:
    3970
    5260
    5263
    5271
    3970
    5260
    5263
    5271
    
    upload:
    94989
    3392
    225370
    
1000:
    7866
    7629
    7925
    7910
    7934
    7619
    8151
    
    upload:
    194819 ms
    196446
    188416
"""


def print_mean():
    print("100:", np.mean([798,1243, 1248,1248,1254,1906,1933,905,881,807,1194,1240,1239,1245]))
    print("500:", np.mean([3970,5260,5263,5271,3970,5260,5263,5271]))
    print("1000:", np.mean([7866,7629,7925,7910,7934,7619]))
    print("100:", np.mean([74608, 74798, 74812, 75695, 21323]))
    print("500:", np.mean([94989, 3392, 225370]))
    print("1000:", np.mean([196446, 194819, 188416]))
    """
    zip:
    100: 1224.36
    500: 4941.0
    1000: 7813.83
    
    upload:
    64247.2
    107917.0
    193227.0
    """


# noinspection PyAttributeOutsideInit
class ZipTest(QtWidgets.QWidget):

    signal_update_progress = pyqtSignal(int, int)
    image_queue = Queue()  # (maxsize=128)
    upload_queue = Queue()

    def __init__(self, upload_callback, default_log_folder="zip_test_data"):
        super(ZipTest, self).__init__()
        self.all_images_count = 0
        self.num_transferred_images = 0
        self.num_transferred_folders = 0

        self.zip_time = 0
        self.upload_time = 0
        self.batch_size = 1000  # the number of images per subfolder

        """
        100:
            Original Size: 6.23MB
            New Size: 4.68MB
        
        250:
            Original Size: 13.75MB
            New Size: 10.39MB
        
        500:
            Original Size: 29.74MB
            New Size: 22.12MB
            
        1000:
            Original Size: 60.05MB
            New Size: 45.52MB
            
        1500:
            Original Size: 86.40MB
            New Size: 68.62MB
        """

        self.__upload_callback = upload_callback
        self.__log_folder = default_log_folder

        self.__log_folder_path = pathlib.Path(__file__).parent / self.__log_folder  # creates base folder in "tracking/"
        self.__images_path = self.__log_folder_path / "images"  # tracking/tracking_data/images/
        self.__images_zipped_path = self.__log_folder_path / "images_zipped"  # tracking/tracking_data/images_zipped/
        self.__init_log()

        # TODO use real ones for test:
        self.__hostname = ""
        self.__username = ""
        self.__password = ""
        self.__port = 0000
        # NOTE: setting hostkeys to None is highly discouraged as we lose any protection
        # against man-in-the-middle attacks; however, it is the easiest solution and for now it should be fine
        self.__cnopts = pysftp.CnOpts()
        self.__cnopts.hostkeys = None

        self.__init_server_connection()

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
                    self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)

                    # check if the current number of saved images is a multiple of the batch size
                    if (self.all_images_count % self.batch_size) == 0:
                        # a batch of images is finished so we put this one in a queue to be zipped and uploaded
                        # self.upload_queue.put(str(self.folder_count))

                        # and create a new folder for the next batch
                        self.folder_count += 1
                        # self.signal_update_progress.emit(self.num_transferred_folders, self.folder_count)
                        self.__get_curr_image_folder().mkdir()

            time.sleep(0.030)  # wait for 30 ms as this is the maximum we can get new frames from our webcam

    def start_async_upload(self):
        # connect the custom signal to the callback function to update the gui (updating the gui MUST be done from the
        # main thread and not from a background thread, otherwise it would just randomly crash after some time!!)
        self.signal_update_progress.connect(self.__upload_callback)

        # must not be a daemon thread here!
        self.upload_thread = threading.Thread(target=self.__start_ftp_transfer_alt, name="UploadThread", daemon=False)
        self.upload_thread.start()

    """
    def __start_ftp_transfer(self):
        while self.tracking_active:
            if self.upload_queue.qsize() > 0:
                file_name = self.upload_queue.get()

                # zip the the folder with the given name
                # TODO execute this zipping part in a processpoolexecutor and await result?
                start_time = time.time()
                zip_file_name = self.__zip_folder(file_name)
                end_time = time.time()
                needed_time_z = (end_time - start_time) * 1000
                print(f"zip_folder => {needed_time_z} ms")
                self.zip_time += needed_time_z

                # upload this folder to the server
                start_time = time.time()
                self.__upload_zipped_images(zip_file_name)
                end_time = time.time()
                needed_time_u = (end_time - start_time) * 1000
                print(f"upload_zipped_images => {needed_time_u} ms")
                self.upload_time += needed_time_u

                # update progressbar in gui
                self.num_transferred_folders += 1
                self.num_transferred_images += self.batch_size
                self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
                # self.signal_update_progress.emit(self.num_transferred_folders, self.folder_count)
            
            time.sleep(0.03)  # wait for the same amount of time as the other queue
    """

    def __start_ftp_transfer_alt(self):
        while self.tracking_active:
            num_finished_folders = self.folder_count - 1
            all_img_folders = os.listdir(f"{self.__images_path}")

            if self.num_transferred_folders > num_finished_folders:
                sys.stderr.write("This should not be happening! :(")
                continue

            folders_not_transferred = all_img_folders[self.num_transferred_folders:num_finished_folders]

            """
            for folder_name in folders_not_transferred:
                with ThreadPoolExecutor() as executor:
                    executor.submit(self.__zip_and_upload, folder_name)

                self.num_transferred_folders += 1
                self.num_transferred_images += self.batch_size
                self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)
            
            """
            with concurrent.futures.ThreadPoolExecutor() as texec:
                tasks = [texec.submit(self.__zip_and_upload, folder_name).add_done_callback(self._update_progress)
                         for folder_name in folders_not_transferred]
                # for task in concurrent.futures.as_completed(tasks):
                #     if not task.result():
                #         raise Exception("Failed to upload file.")

            """
            with concurrent.futures.ThreadPoolExecutor() as executor:
                tasks = [executor.submit(self.__zip_and_upload, folder_name) for folder_name in folders_not_transferred]
                for f in tasks:
                    f.add_done_callback(self._update_progress)
                #done, not_done = concurrent.futures.wait(tasks, return_when=concurrent.futures.ALL_COMPLETED)
                #if len(not_done) > 0:
                #    raise Exception("Extraction error.")
            """

    def __zip_and_upload(self, folder_name: str):
        zip_file_name = self.__zip_folder(folder_name)
        # upload this folder to the server
        self.__upload_zipped_images(zip_file_name)

    def _update_progress(self, future):
        print("done: ", future.done())
        self.num_transferred_folders += 1
        self.num_transferred_images += self.batch_size
        self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)

    @timeit
    def __zip_folder(self, folder_name: str):
        """
        Converts the given folder to a 7zip archive file to speed up the upload.
        """
        folder_path = f"{self.__images_path / folder_name}"
        zip_file_name = folder_name + ".7z"

        # zip_filters = [{'id': FILTER_X86}, {'id': FILTER_LZMA2, 'preset': PRESET_DEFAULT}]  # default
        # zip_filters = [{'id': FILTER_DELTA}, {'id': FILTER_LZMA2, 'preset': PRESET_DEFAULT}]
        # zip_filters = [{'id': FILTER_ZSTD, 'level': 3}]
        # zip_filters = [{'id': FILTER_LZMA2, 'preset': PRESET_DEFAULT}]
        zip_filters = [{'id': FILTER_BROTLI, 'level': 3}]
        # zip_filters = [{'id': FILTER_BZIP2}]
        # zip_filters = [{'id': FILTER_LZMA}]

        with py7zr.SevenZipFile(f"{self.__images_zipped_path / zip_file_name}", 'w', filters=zip_filters) as archive:
            archive.writeall(folder_path)
        return zip_file_name

    """
    0) Default LZMA2 + BCJ:
    100 batch size:
    Time needed to upload 100 images: 11.709 seconds
    Average zip time over 2 folders: 3028.41 ms
    Average upload time over 2 folders: 4688.00 ms
    Average improvement over 3 folders: 2.16MB

    500:
    Average zip time over 2 folders: 15361.71 ms
    Average upload time over 2 folders: 19001.10 ms
    Average improvement over 3 folders: 11.90MB

    1) LZMA:
    Time needed to upload 100 images: 11.369 seconds
    Average zip time over 2 folders: 3141.26 ms
    Average upload time over 2 folders: 3318.36 ms
    Average improvement over 3 folders: 2.34MB

    1.5) LZMA2 and Preset Default:
    Time needed to upload 100 images: 11.520 seconds
    Average zip time over 2 folders: 2879.07 ms
    Average upload time over 2 folders: 3391.28 ms
    Average improvement over 3 folders: 2.12MB

    500:
    Average zip time over 2 folders: 14360.78 ms
    Average upload time over 2 folders: 20129.70 ms
    Average improvement over 3 folders: 12.01MB

    2) BZIP2:
    Time needed to upload 100 images: 11.364 seconds
    Average zip time over 2 folders: 1959.20 ms
    Average upload time over 2 folders: 4085.70 ms
    Average improvement over 3 folders: 2.15MB

    500:
    Average zip time over 2 folders: 8140.12 ms
    Average upload time over 2 folders: 22821.38 ms
    Average improvement over 3 folders: 10.38MB

    3) BROTLI level 3:
    Time needed to upload 100 images: 9.891 seconds
    Average zip time over 2 folders: 1280.65 ms
    Average upload time over 2 folders: 3446.01 ms
    Average improvement over 3 folders: 1.99MB

    500:
    Average zip time over 2 folders: 6126.30 ms
    Average upload time over 2 folders: 23804.95 ms
    Average improvement over 3 folders: 11.44MB

    4) ZSTD level 3
    Time needed to upload 100 images: 10.056 seconds
    Average zip time over 2 folders: 1166.41 ms
    Average upload time over 2 folders: 4334.71 ms
    Average improvement over 3 folders: 1.72MB

    5) LZMA2 + Delta
    Time needed to upload 100 images: 11.705 seconds
    Average zip time over 2 folders: 3063.91 ms
    Average upload time over 2 folders: 3383.97 ms
    Average improvement over 3 folders: 2.02MB

    500:
    Average zip time over 3 folders: 14802.33 ms
    Average upload time over 3 folders: 22318.45 ms
    Average improvement over 3 folders: 5.97MB
    """

    @timeit
    def __upload_zipped_images(self, file_name):
        try:
            with pysftp.Connection(host=self.__hostname,
                                   username=self.__username,
                                   password=self.__password, port=self.__port, cnopts=self.__cnopts) as sftp:
                sftp.put(localpath=f"{self.__images_zipped_path / file_name}",
                         remotepath=f"{self.user_dir}/images/{file_name}")
        except Exception as e:
            sys.stderr.write(f"Exception during image upload occurred: {e}")

    def finish_logging(self):
        # when tracking is stopped the current batch of images will never reach the upload condition, so we do it now
        # self.upload_queue.put(str(self.folder_count))
        self.folder_count += 1

    def stop_upload(self):
        self.tracking_active = False
        # use join to wait for thread finish as this is no daemon thread (this blocks main ui!)
        # self.upload_thread.join()

        # close the connection to the sftp server
        self.sftp.close()

        print(3 * "###")
        print(f"\nAverage zip time over {self.num_transferred_folders} folders: "
              f"{self.zip_time / self.num_transferred_folders:.2f} ms")
        print(f"\nAverage upload time over {self.num_transferred_folders} folders: "
              f"{self.upload_time / self.num_transferred_folders:.2f} ms")
        print(3 * "###")

        # clear the image queue in a thread safe way
        with self.image_queue.mutex:
            self.image_queue.queue.clear()
        with self.upload_queue.mutex:
            self.upload_queue.queue.clear()

        # remove local dir after uploading everything
        # shutil.rmtree(self.__images_path)


class ZipMain(QtWidgets.QWidget):

    def __init__(self, debug_active=True):
        super(ZipMain, self).__init__()
        self.debug = debug_active

        self.__stop = False
        self.__setup_gui()
        self.__init_logger()
        self.fps_measurer = FpsMeasurer()

    def __setup_gui(self):
        self.setGeometry(50, 50, 600, 200)
        self.setWindowTitle("Upload Progress")
        self.layout = QtWidgets.QVBoxLayout()

        # show some instructions
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Press Ctrl + Shift + A to start tracking and Ctrl + Shift + Q to stop it (once stopped it "
                           "cannot be restarted with the hotkey!).\n\nPlease don't close this window until the upload"
                           " is finished!\nThis may take some time depending on your hardware and internet connection.")
        self.label.setContentsMargins(10, 10, 10, 10)
        self.label.setStyleSheet("QLabel {font-size: 9pt;}")
        self.layout.addWidget(self.label)

        self.__setup_progress_bar()

        # show status of tracking
        self.tracking_status = QtWidgets.QLabel(self)
        self.tracking_status.setContentsMargins(0, 10, 0, 10)
        self.tracking_status.setAlignment(Qt.AlignCenter)
        self.__set_tracking_status_ui()
        self.layout.addWidget(self.tracking_status)

        # add buttons to manually start and stop tracking
        self.__setup_button_layout()  # TODO only for debugging!

        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

    def __setup_button_layout(self):
        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setText("Start tracking")
        self.start_button.setStyleSheet("QPushButton {background-color: rgb(87, 205, 0); color: black; "
                                        "padding: 10px 10px 10px 10px; border-radius: 2px;}")
        self.start_button.clicked.connect(self.start)  # connect start method to this button

        self.stop_button = QtWidgets.QPushButton(self)
        self.stop_button.setText("Stop tracking")
        self.stop_button.setStyleSheet("QPushButton {background-color: rgb(153, 25, 25); color: white; "
                                       "padding: 10px 10px 10px 10px; border-radius: 2px;}")
        self.stop_button.clicked.connect(self.stop)  # connect stop method to this button

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.stop_button, alignment=Qt.AlignRight)
        self.layout.addLayout(button_layout)

    def __setup_progress_bar(self):
        # show a progressbar for the upload
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        progress_bar_layout = QtWidgets.QHBoxLayout()
        progress_bar_layout.addStretch(1)
        progress_bar_layout.addWidget(self.progress_bar, stretch=7)
        progress_bar_layout.addStretch(1)
        self.layout.addLayout(progress_bar_layout)

        # show how much files have been uploaded
        self.label_current = QtWidgets.QLabel(self)
        self.label_all = QtWidgets.QLabel(self)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.label_current)
        hbox.addWidget(self.label_all)
        hbox.setAlignment(Qt.AlignCenter)
        hbox.setSpacing(5)
        self.layout.addLayout(hbox)

        # show approximate time for the upload
        self.label_eta = QtWidgets.QLabel(self)
        self.label_eta.setAlignment(Qt.AlignCenter)
        self.label_eta.setStyleSheet("font-weight: bold")
        self.layout.addWidget(self.label_eta)

    def __set_tracking_status_ui(self):
        if not self.__stop:
            self.tracking_status.setText("Tracking active")
            self.tracking_status.setStyleSheet("QLabel {color: green;}")
        else:
            self.tracking_status.setText("Tracking not active")
            self.tracking_status.setStyleSheet("QLabel {color: red;}")

    def __init_logger(self):
        self.__logger = ZipTest(self.__on_upload_progress)

    def __on_upload_progress(self, current, overall):
        if overall == 0 or current > overall:
            return

        if current != 0:
            seconds_per_frame = (time.time() - self.__upload_start) / current
            eta_seconds = (overall - current) * seconds_per_frame
            minutes = math.floor(eta_seconds / 60)
            seconds = round(eta_seconds % 60)
            self.label_eta.setText(f"Remaining Upload Time: {minutes} min, {seconds} seconds")
        else:
            self.label_eta.setText(f"Waiting for more data...")

        self.label_current.setText(str(current))
        self.label_all.setText(f"/ {overall}")
        progress = (current / overall) * 100

        # TODO remove later:
        if current in [30, 100, 1200]:
            needed_time = time.time() - self.__upload_start
            # print(f"Time needed to upload {current} images: {needed_time:.3f} seconds")

        self.progress_bar.setValue(int(progress))

    def start(self):
        self.tracking_thread = threading.Thread(target=self.__start_tracking, name="TrackingThread", daemon=True)
        self.tracking_thread.start()

    def __start_tracking(self):
        self.__logger.start_saving_images_to_disk()  # start saving webcam frames to disk
        self.__logger.start_async_upload()  # start uploading data to sftp server

        self.__upload_start = time.time()
        if self.debug:
            self.fps_measurer.start()

        images_dir = os.listdir("../zip_test_images/")
        for image in images_dir:
            frame = cv2.imread(f"../zip_test_images/{image}")
            if frame is None or self.__stop:
                break

            processed_frame = self.__process_frame(frame)
            self.__current_frame = processed_frame
            self.fps_measurer.update()

            time.sleep(0.03)

        self.__logger.finish_logging()

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        log_timestamp = get_timestamp()
        self.__logger.add_image_to_queue("capture", frame, log_timestamp)
        return frame

    def stop(self):
        self.__stop = True
        self.__set_tracking_status_ui()

        self.__logger.finish_logging()

        if self.debug:
            self.fps_measurer.stop()
            print(f"[INFO] elapsed time: {self.fps_measurer.elapsed():.2f} seconds")

    def closeEvent(self, event) -> None:
        self.__logger.stop_upload()

        improvement = 0
        c = 0
        for idx, folder in enumerate(os.listdir("zip_test_data/images_zipped")):
            dir_size_new = get_directory_size(f"zip_test_data/images_zipped/{folder}")
            new_size = get_size_format(dir_size_new)

            normal_folder = os.listdir("zip_test_data/images")[idx]
            dir_size_original = get_directory_size(f"zip_test_data/images/{normal_folder}")
            original_size = get_size_format(dir_size_original)

            improvement += dir_size_original - dir_size_new
            c += 1

            print(5*"###")
            print(f"Original Size: {original_size}\nNew Size: {new_size}")
            print(5 * "###")

        print(f"\nAverage improvement over {c} folders: {get_size_format(improvement/c)}")

        event.accept()


def main():
    print_mean()

    app = QApplication(sys.argv)
    zt = ZipMain()
    zt.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
