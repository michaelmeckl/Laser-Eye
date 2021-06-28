import gzip
import io
import shutil
import sys
import threading
import time
import zlib
import zipfile
import schedule
from post_processing.image_utils import encode_image


def __start_scheduling(self):
    """
    Starts a background thread that logs the currently saved data (a list of dicts) to the csv file;
    this is far more efficient than appending the log data row by row as this actually re-creates the
    whole pandas dataframe completely every time;
    see https://stackoverflow.com/questions/10715965/create-pandas-dataframe-by-appending-one-row-at-a-time

    Another benefit is that it moves the I/O operation (writing to file) happen less often and off the
    main thread.


    Cancel with:
        # cancel current scheduling job
        active_jobs = schedule.get_jobs(self.__log_tag)
        if len(active_jobs) > 0:
            schedule.cancel_job(active_jobs[0])
        # Stop the background thread on the next schedule interval
        self.__logging_job.set()
    """
    schedule_interval = 3  # schedule logging to csv file periodically
    schedule.every(schedule_interval).seconds.do(self.__save_images).tag(self.__log_tag)
    # Start the background thread
    self.__logging_job = run_continuously()


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


def _upload_image_zipped(self, image):
    with open(f"{self.__images_file_path}/{image}", 'rb') as f_in:
        with gzip.open(f"{self.__images_file_path}/{image}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            self.sftp.put(localpath=f"{self.__images_file_path}/{image}.gz",
                          remotepath=f"{self.user_dir}/images/{image}.gz")
            self.num_transferred_images += 1
            self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)


def _zip_try_3(self, images_not_transferred: list):
    with zipfile.ZipFile(f'{self.__images_file_path}/sample2.zip', 'w') as zipObj2:
        for image in images_not_transferred:
            zipObj2.write(f'{self.__images_file_path}/{image}')
        self.sftp.put_d(f'{self.__images_file_path}\\sample2.zip', remotepath=f"{self.user_dir}/images/")
        self.num_transferred_images += len(images_not_transferred)
        self.signal_update_progress.emit(self.num_transferred_images, self.all_images_count)


def zip_folder(self, dir_name):
    shutil.make_archive(f"{dir_name}_output", 'zip', dir_name)
    """
    filePaths = []
    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(dir_name):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)

    # writing files to a zipfile
    zip_file = zipfile.ZipFile(dir_name + '.zip', 'w')
    with zip_file:
        # writing each file one by one
        for file in filePaths:
            zip_file.write(file)
    """


def _compress_gzip(self, all_images_count):
    # not working
    filename, image, timestamp = self.image_queue.get()
    byte_stream = encode_image(image, img_format=".jpeg")
    # s_out = gzip.compress(bytes(image), compresslevel=9)
    g = gzip.GzipFile(filename=f"{filename}.jpeg", fileobj=byte_stream,
                      compresslevel=7, mode='rb')
    self.sftp.putfo(g, f"{self.user_dir}/images/{filename}__{timestamp}")
    self.num_transferred_images += 1
    self.signal_update_progress.emit(self.num_transferred_images, all_images_count)


def _compress_zlib(self, all_images_count):
    filename, image, timestamp = self.image_queue.get()
    byte_stream = encode_image(image, img_format=".jpeg")
    compressed_bt = zlib.compress(byte_stream.getvalue(), level=5)  # 0 is lowest compression, 9 highest (9 takes
    # too long)
    self.sftp.putfo(io.BytesIO(compressed_bt), f"{self.user_dir}/images/{filename}__{timestamp}")
    self.num_transferred_images += 1
    self.signal_update_progress.emit(self.num_transferred_images, all_images_count)


def __start_ftp_transfer_byte_version(self):
    while self.tracking_active:
        all_images_count = self.image_queue.qsize()

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
