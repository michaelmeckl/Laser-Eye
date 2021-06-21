#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import io
import math
import time
from datetime import datetime
import platform
import sys
import threading
import cv2
import dlib
import numpy as np
import psutil
import pyautogui
from PIL import Image
from plyer import notification
from post_processing.image_utils import scale_image, resize_image, preprocess_frame, extract_image_region
from service.face_detector import MxnetDetectionModel
from tracking.ThreadedWebcamCapture import WebcamStream
from tracking.TrackingLogger import Logger, TrackingData, get_timestamp
from tracking.FpsMeasuring import FpsMeasurer
# from pynput import keyboard
import keyboard
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox


# TODO:
# Warnings mit einbauen, wenn der Nutzer sich zu viel bewegt oder daten zu schlecht
# und die Zeitpunkte hier mitloggen!

# with title = pyautogui.getActiveWindow().title we could log the title of the current game if necessary


# class TrackingSystem(QtWidgets.QDialog):
class TrackingSystem(QtWidgets.QMainWindow):

    def __init__(self, capturing_device):
        super(TrackingSystem, self).__init__()
        self.__current_frame = None
        self.__tracking_active = False

        self.frame_count = 0
        self.t1 = None
        self.t2 = None

        self.capture = capturing_device

        self.hog_detector = dlib.get_frontal_face_detector()
        self.face_detector = MxnetDetectionModel("weights/16and32", 0, .6, gpu=-1)

        self.__init_gui()
        self.__init_logger()

        # TODO only for debugging! -> disable fps measuring later in the entire class
        self.fps_measurer = FpsMeasurer()
        capture_fps = self.capture.get_stream_fps()
        self.fps_measurer.show_optimal_fps(capture_fps)

    def __init_logger(self):
        self.__logger = Logger(self.__on_upload_progress)
        self.__tracked_data = {key.name: None for key in TrackingData}
        self.__log_static_data()

    def __init_gui(self):
        self.setGeometry(50, 50, 600, 200)
        self.setWindowTitle("Upload Progress")

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(50, 10, 550, 20)
        self.label.setText("Uploading webcam images... Please don't close until finished!")

        self.progress_bar_overall = QtWidgets.QProgressBar(self)
        self.progress_bar_overall.setGeometry(50, 40, 550, 40)
        self.progress_bar_overall.setMaximum(100)

        self.label_current = QtWidgets.QLabel(self)
        self.label_current.setGeometry(200, 80, 60, 20)
        self.label_all = QtWidgets.QLabel(self)
        self.label_all.setGeometry(265, 80, 60, 20)

        self.label_eta = QtWidgets.QLabel(self)
        self.label_eta.setGeometry(220, 120, 250, 20)

    def closeEvent(self, event):
        choice = QMessageBox.question(self, 'Warning', "Please close this window only if all images have been "
                                                       "transferred! Do you really want to exit? ",
                                      QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            # self.close()
            event.accept()
        else:
            event.ignore()

    # TODO ca 5 sec für 30 bilder  -> 0.16 sec pro bild (ca. 160 ms)
    # TODO # aktuell in etwa 150 ms pro Frame
    #  11 min für 5200 Frames -> 50min * 60 * 25fps = 75000 frames -> 158 min
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

        if current in [30, 167]:
            needed_time = time.time() - self.__upload_start
            print(f"Time needed to upload {current} images: {needed_time:.3f} seconds")

        self.progress_bar_overall.setValue(int(progress))

    def listen_for_hotkey(self, hotkey_toggle="ctrl+shift+a", hotkey_stop="ctrl+shift+q"):
        # keyboard module
        keyboard.add_hotkey(hotkey_toggle, self.__toggle_tracking_status, suppress=False, trigger_on_release=False)
        keyboard.add_hotkey(hotkey_stop, self.__stop_tracking, suppress=False, trigger_on_release=False)

        # pynput module
        # listener = keyboard.Listener( on_press=on_press, on_release=on_release)
        # listener.start()  # only works on background thread!! otherwise finishes immediately
        # listener.join()  # use join after while loop or in break!!!!

    # TODO atm this finished the main thread as well!  -> while loop in main after listen as well?
    def __toggle_tracking_status(self):
        # toggle tracking on hotkey press
        if self.__tracking_active:
            # TODO don't allow to stop tracking with hotkey to prevent errors? (only via GUI or automatically after
            #  download finished?
            self.__tracking_active = False
            self.__stop_tracking()
            notification.notify(title="Tracking stopped", timeout=1)
        else:
            # start tracking on a background thread
            self.__tracking_active = True
            notification.notify(title="Tracking started", timeout=1)
            self.capture.start()  # start reading frames from webcam
            self.tracking_thread = threading.Thread(target=self.__start_tracking, daemon=True)
            self.tracking_thread.start()

    def __stop_tracking(self):
        """
        Stop and cleanup active video/webcam captures and destroy open windows if any.
        Also stop the logging.
        """
        self.__tracking_active = False
        self.capture.stop()
        cv2.destroyAllWindows()
        self.__logger.stop_scheduling()

        self.fps_measurer.stop()
        print(f"[INFO] elapsed time: {self.fps_measurer.elapsed():.2f} seconds")
        print(f"[INFO] approx. FPS on background thread: {self.fps_measurer.fps():.2f}")
        print("Frames on main thread:", self.fps_measurer._numFrames)

    def __start_tracking(self):
        """
        This function runs on a background thread.
        """
        self.__logger.start_async_upload()  # start uploading data

        self.__upload_start = time.time()
        self.fps_measurer.start()
        while True:
            if not self.__tracking_active:
                break

            frame = self.capture.read()
            if frame is None:
                sys.stderr.write("Frame from stream thread is None! This shouldn't happen!")
                break

            # self.__measure_frame_count()  # TODO only for debugging
            # update the FPS counter
            self.fps_measurer.update()

            processed_frame = self.__process_frame(frame)
            if processed_frame is None:
                continue
            self.__current_frame = processed_frame

            cv2.putText(processed_frame, f"current FPS: {self.fps_measurer.get_current_fps():.3f}",
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("current_frame", processed_frame)
            # read until the video is finished or if no video was provided until the
            # user presses 'q' on the keyboard;
            # replace 1 with 0 to step manually through the video "frame-by-frame"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # cleanup after while loop
        self.__tracking_active = False
        # TODO
        # self.__stop_tracking()

    def __measure_frame_count(self):
        self.frame_count += 1
        print(f"########\nFrame {self.frame_count} at {datetime.now()}\n#######")
        if self.frame_count % 2 == 1:
            self.t1 = get_timestamp()
        elif self.frame_count % 2 == 0:
            self.t2 = get_timestamp()
            print(f"########\nTime between frames {(self.t2 - self.t1):.2f} seconds\n#######")

    def downsample(self, frame):
        scaled = scale_image(frame, scale_factor=0.5, show_scaled=True)
        print(f"Shape_scaled: {scaled.shape[0]}, {scaled.shape[1]}")
        resized = resize_image(frame, size=300, show_resized=True)
        print(f"Shape_resized: {resized.shape[0]}, {resized.shape[1]}")
        return scaled

    def to_gray(self, frame):
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return grayscale_image

    def find_face_hog(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # dlib requires RGB while opencv uses BGR per default
        bboxes = self.hog_detector(rgb_frame, 0)  # 0 so it won't be upsampled
        if len(bboxes) > 0:
            # only take the first face if more were found (in most cases there should be only one anyway)
            face = bboxes[0]
            region = extract_image_region(frame, face.left(), face.top(), face.right(), face.bottom())
            cv2.imshow("extracted_region_mxnet", region)
            return region

    def find_face_mxnet(self, frame):
        bboxes = self.face_detector.detect(frame)  # TODO if this is used bboxes should be logged as well?
        face_region = None
        for face in bboxes:
            face_region = extract_image_region(frame, face[0], face[1], face[2], face[3])
            cv2.imshow("extracted_region_mxnet", face_region)
            break  # break to take only the first face (in most cases there should be only one anyway)
        return face_region

    def find_face_mxnet_resized(self, frame, inHeight=300, inWidth=0):
        image_frame = frame.copy()
        frameHeight = image_frame.shape[0]
        frameWidth = image_frame.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)
        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        image_frame_small = cv2.resize(image_frame, (inWidth, inHeight))
        # image_frame_small = cv2.cvtColor(image_frame_small, cv2.COLOR_BGR2RGB)
        bboxes = self.face_detector.detect(image_frame_small)
        face_region = None
        for face in bboxes:
            face_region = extract_image_region(frame,
                                               face[0] * scaleWidth,
                                               face[1] * scaleHeight,
                                               face[2] * scaleWidth,
                                               face[3] * scaleHeight)
            cv2.imshow("extracted_region_mxnet", face_region)
            break  # break to take only the first face (in most cases there should be only one anyway)
        return face_region

    """
    30 Frames dauern aktuell in etwa 5 sek; ca. 25 fps mit facedetect;
    wenn ich alle 20 sekunden schedule, die aktuellen bilder zu nehmen würde das für das erste mal bedeuten:
    20*25 = 500 frames -> 500 / 30 * 16 -> 5 * 16 = 80sec  (Baseline)
    TODO: zip versuchen!
    """
    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        face_image = self.find_face_mxnet(frame)
        # face_image = self.find_face_mxnet_resized(frame)  # ca. 2 seconds faster?
        # face_image = self.find_face_hog(frame)
        # new_image = self.downsample(frame)
        # new_image = self.to_gray(face_image)

        if face_image is not None:
            # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
            # can't be matched later!
            log_timestamp = get_timestamp()
            self.__logger.log_image("capture", face_image, log_timestamp)
            # self.__logger.add_image_to_queue("capture", face_image, log_timestamp)
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
            # **platform.uname()._asdict(),
            # **psutil.virtual_memory()._asdict(),
        })
        self.__logger.log_static_data(data=self.__tracked_data)

    def get_current_frame(self) -> np.ndarray:
        return self.__current_frame

    def debug_start_tracking(self):
        # TODO only for faster debugging; remove later
        # start tracking on a background thread
        self.__tracking_active = True
        self.capture.start()  # start reading frames from webcam
        self.tracking_thread = threading.Thread(target=self.__start_tracking, daemon=True)
        self.tracking_thread.start()


def main():
    # use a custom threaded video capture to increase fps;
    # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    # TODO test if using transform in the webcam thread like in file video capture would speed up things !
    capture = WebcamStream(src=0)

    app = QApplication(sys.argv)
    tracking_system = TrackingSystem(capture)
    tracking_system.show()
    tracking_system.listen_for_hotkey()

    print("Press ctrl + shift + a to toggle tracking and ctrl + shift + q to stop it!")
    if args.exec:
        tracking_system.debug_start_tracking()

    # TODO only for debugging:
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

    sys.exit(app.exec_())


if __name__ == "__main__":
    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Webcam eye tracking system that logs webcam images to an sftp "
                                                 "server.")
    parser.add_argument("-x", "--exec", help="Starts tracking immediately.", action="store_true")
    args = parser.parse_args()

    main()
