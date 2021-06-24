#!/usr/bin/python3
# -*- coding:utf-8 -*-

import math
import time
from datetime import datetime
import platform
import sys
import threading
from pathlib import Path
from tracking_utils import scale_image, find_face_mxnet
import cv2
import numpy as np
import psutil
import pyautogui
from PyQt5.QtCore import Qt
from plyer import notification
from tracking_service.face_detector import MxnetDetectionModel
from ThreadedWebcamCapture import WebcamStream
from TrackingLogger import Logger, TrackingData, get_timestamp
from FpsMeasuring import FpsMeasurer
import keyboard
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox


# TODO ca 5 sec für 30 bilder  -> 0.16 sec pro bild (ca. 160 ms)
# TODO # aktuell in etwa 150 ms pro Frame
#  11 min für 5200 Frames -> 50min * 60 * 25fps = 75000 frames -> 158 min


# FIXME: tracking can be restarted (via button and shortcut) after being stopped which completely destroys everything
# -> simply unregister callbacks and disconnect slots in stop ?


class TrackingSystem(QtWidgets.QWidget):

    def __init__(self, debug_active=True):
        super(TrackingSystem, self).__init__()
        self.__tracking_active = False
        self.debug = debug_active

        self.__current_frame = None
        self.frame_count = 0
        self.t1 = None
        self.t2 = None

        # use a custom threaded video capture to increase fps;
        # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
        self.capture = WebcamStream(src=0)

        # necessary for building the exe file with pyinstaller with the --one-file option as the path changes;
        # see https://stackoverflow.com/questions/7674790/bundling-data-files-with-pyinstaller-onefile for more
        if getattr(sys, 'frozen', False):
            # change the path if we are executing the exe file; the lint warning here can be ignored ;)
            folder = Path(sys._MEIPASS)
            data_path = folder/'weights/16and32'
        else:
            folder = Path(__file__).parent
            data_path = folder/'../weights/16and32'

        self.face_detector = MxnetDetectionModel(data_path, 0, .6, gpu=-1)

        self.__setup_gui()
        self.__init_logger()

        if self.debug:
            self.fps_measurer = FpsMeasurer()
            capture_fps = self.capture.get_stream_fps()
            self.fps_measurer.show_optimal_fps(capture_fps)

    def __setup_gui(self):
        self.setGeometry(50, 50, 600, 200)
        self.setWindowTitle("Upload Progress")
        self.layout = QtWidgets.QVBoxLayout()

        # show some instructions
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Press Ctrl + Shift + A to start tracking and Ctrl + Shift + Q to stop it (or use the "
                           "button below).\n\nPlease don't close this window until the upload is finished!\n"
                           "This may take some time depending on your hardware and internet connection.")
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
        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setText("Start tracking")
        self.start_button.setStyleSheet("QPushButton {background-color: rgb(87, 205, 0); color: black; "
                                        "padding: 10px 10px 10px 10px; border-radius: 2px;}")
        self.start_button.clicked.connect(self.__activate_tracking)

        self.stop_button = QtWidgets.QPushButton(self)
        self.stop_button.setText("Stop tracking")
        self.stop_button.setStyleSheet("QPushButton {background-color: rgb(153, 25, 25); color: white; "
                                       "padding: 10px 10px 10px 10px; border-radius: 2px;}")
        self.stop_button.clicked.connect(self.__stop_tracking)  # connect stop method to this button

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.start_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(self.stop_button, alignment=Qt.AlignRight)
        self.layout.addLayout(button_layout)

        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

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
        if self.__tracking_active:
            self.tracking_status.setText("Tracking active")
            self.tracking_status.setStyleSheet("QLabel {color: green;}")
        else:
            self.tracking_status.setText("Tracking not active")
            self.tracking_status.setStyleSheet("QLabel {color: red;}")

    def __init_logger(self):
        self.__logger = Logger(self.__on_upload_progress)
        self.__tracked_data = {key.name: None for key in TrackingData}
        self.__log_static_data()

    def __on_upload_progress(self, current, overall):
        if overall == 0 or current > overall:
            return

        seconds_per_frame = (time.time() - self.__upload_start) / current
        eta_seconds = (overall - current) * seconds_per_frame
        minutes = math.floor(eta_seconds / 60)
        seconds = round(eta_seconds % 60)
        self.label_eta.setText(f"ETA: {minutes} min, {seconds} seconds")
        self.label_current.setText(str(current))
        self.label_all.setText(f"/ {overall}")
        progress = (current / overall) * 100

        # TODO remove later:
        if current in [30, 167]:
            needed_time = time.time() - self.__upload_start
            print(f"Time needed to upload {current} images: {needed_time:.3f} seconds")

        self.progress_bar.setValue(int(progress))

    def listen_for_hotkey(self, hotkey_start="ctrl+shift+a", hotkey_stop="ctrl+shift+q"):
        keyboard.add_hotkey(hotkey_start, self.__activate_tracking, suppress=False, trigger_on_release=False)
        keyboard.add_hotkey(hotkey_stop, self.__stop_tracking, suppress=False, trigger_on_release=False)

    def __activate_tracking(self):
        # activate tracking on hotkey press
        if not self.__tracking_active:
            # start tracking on a background thread
            self.__tracking_active = True
            self.__set_tracking_status_ui()
            notification.notify(title="Tracking started", message="Tracking is now active!", timeout=1)
            self.capture.start()  # start reading frames from webcam
            self.tracking_thread = threading.Thread(target=self.__start_tracking, daemon=True)
            self.tracking_thread.start()
        else:
            notification.notify(title="Can't start tracking!", message="Tracking is already active!", timeout=2)

    def __start_tracking(self):
        """
        This function runs on a background thread.
        """
        self.__logger.start_saving_images_to_disk()  # start saving webcam frames to disk
        self.__logger.start_async_upload()  # start uploading data to sftp server

        self.__upload_start = time.time()
        if self.debug:
            self.fps_measurer.start()

        while True:
            if not self.__tracking_active:
                break

            frame = self.capture.read()
            if frame is None:
                sys.stderr.write("Frame from stream thread is None! This shouldn't happen!")
                break

            processed_frame = self.__process_frame(frame)
            if processed_frame is None:
                continue

            self.__current_frame = processed_frame
            # self.__measure_frame_count()  # TODO delete

            if self.debug:
                self.fps_measurer.update()
                cv2.putText(processed_frame, f"current FPS: {self.fps_measurer.get_current_fps():.3f}",
                            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("current_frame", processed_frame)
                # read until the video is finished or if no video was provided until the
                # user presses 'q' on the keyboard;
                # replace 1 with 0 to step manually through the video "frame-by-frame"
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def __measure_frame_count(self):
        self.frame_count += 1
        print(f"########\nFrame {self.frame_count} at {datetime.now()}\n#######")
        if self.frame_count % 2 == 1:
            self.t1 = get_timestamp()
        elif self.frame_count % 2 == 0:
            self.t2 = get_timestamp()
            print(f"########\nTime between frames {(self.t2 - self.t1):.2f} seconds\n#######")

    # TODO remove?
    def downsample(self, frame):
        scaled = scale_image(frame, scale_factor=0.5, show_scaled=True)
        print(f"Shape_scaled: {scaled.shape[0]}, {scaled.shape[1]}")
        # resized = resize_image(frame, size=300, show_resized=True)
        # print(f"Shape_resized: {resized.shape[0]}, {resized.shape[1]}")
        return scaled

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        face_image = find_face_mxnet(self.face_detector, frame)  # TODO if this is used bboxes should be logged as well?
        # face_image = find_face_mxnet_resized(self.face_detector, frame)  # TODO ?
        # new_image = self.downsample(frame)

        if face_image is not None:
            # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
            # can't be matched later!
            log_timestamp = get_timestamp()
            self.__logger.add_image_to_queue("capture", face_image, log_timestamp)
        return frame

    def __log_static_data(self):
        # get the dimensions of the webcam
        video_width, video_height = self.capture.get_stream_dimensions()
        # get the dimensions of the primary monitor.
        screenWidth, screenHeight = pyautogui.size()

        system_info = platform.uname()._asdict()
        ram_info = psutil.virtual_memory()._asdict()

        # noinspection PyProtectedMember
        self.__tracked_data.update({
            TrackingData.SCREEN_WIDTH.name: screenWidth,
            TrackingData.SCREEN_HEIGHT.name: screenHeight,
            TrackingData.CAPTURE_WIDTH.name: video_width,
            TrackingData.CAPTURE_HEIGHT.name: video_height,
            TrackingData.CAPTURE_FPS.name: self.capture.get_stream_fps(),
            TrackingData.CORE_COUNT.name: psutil.cpu_count(logical=True),
            TrackingData.CORE_COUNT_PHYSICAL.name: psutil.cpu_count(logical=False),
            TrackingData.CORE_COUNT_AVAILABLE.name: len(psutil.Process().cpu_affinity()),  # number of usable cpus by
            # this process
            TrackingData.SYSTEM.name: system_info["system"],
            TrackingData.SYSTEM_VERSION.name: system_info["release"],
            TrackingData.MODEL_NAME.name: system_info["node"],
            TrackingData.PROCESSOR.name: system_info["machine"],
            TrackingData.RAM_OVERALL.name: ram_info["total"] / 1000000000,  # convert from Bytes to GB
            TrackingData.RAM_AVAILABLE.name: ram_info["available"] / 1000000000,
            TrackingData.RAM_FREE.name: ram_info["free"] / 1000000000,
        })
        self.__logger.log_csv_data(data=self.__tracked_data)

    def get_current_frame(self) -> np.ndarray:
        return self.__current_frame

    def __stop_tracking(self):
        """
        Stop and cleanup active webcam captures and destroy open windows if any.
        Also stop the logging.
        """
        if self.__tracking_active:
            notification.notify(title="Tracking stopped", message="Tracking has been stopped!", timeout=1)
            self.__tracking_active = False
            self.__set_tracking_status_ui()
            self.capture.stop()
            cv2.destroyAllWindows()

            if self.debug:
                self.fps_measurer.stop()
                print(f"[INFO] elapsed time: {self.fps_measurer.elapsed():.2f} seconds")
                print(f"[INFO] approx. FPS on background thread: {self.fps_measurer.fps():.2f}")
                print("Frames on main thread:", self.fps_measurer._numFrames)

    def closeEvent(self, event):
        choice = QMessageBox.question(self, 'Stop tracking system?',
                                      "Please close this window only if the upload is finished! Do you "
                                      "really want to exit? ",
                                      QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            self.__stop_tracking()
            self.__logger.stop_upload()
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    tracking_system = TrackingSystem()
    tracking_system.show()
    tracking_system.listen_for_hotkey()

    # TODO only for debugging fps:
    """
    c = 0
    start_time = datetime.now()
    while True:
        curr_frame = tracking_system.get_current_frame()
        if curr_frame is None:
            continue
        c += 1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        fps = c / elapsed_time if elapsed_time != 0 else c
        cv2.putText(curr_frame, f"mainthread FPS: {fps:.3f}",
                    (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("fps_main_thread", curr_frame)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break
    """
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


"""
For creating exe with pyinstaller:
add-data C:/Users/Michael/Documents/GitHub/Tracking_System_Version_1.0.1/weights
add-data C:/Users/Michael/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet

hidden-imports: pandas, pysftp and plyer.platforms.win.notification
# for plyer see https://stackoverflow.com/questions/56281839/issue-with-plyer-library-of-python-when-creating-a
-executable-using-pyinstaller


Created pyinstaller command:

pyinstaller --noconfirm --onefile --windowed 
--add-data "C:/Users/Michael/AppData/Local/Programs/Python/Python39/Lib/site-packages/mxnet;mxnet/"
--add-data "C:/Users/Michael/Documents/GitHub/Tracking_System_Version_1.0.1/weights;weights/"
--hidden-import "plyer.platforms.win.notification" --hidden-import "pandas" --hidden-import "pysftp"
"C:/Users/Michael/Documents/GitHub/Tracking_System_Version_1.0.1/tracker.py"
"""
