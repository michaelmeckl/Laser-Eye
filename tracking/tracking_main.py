#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
from datetime import datetime
import platform
import sys
import threading
import cv2
import numpy as np
import psutil
import pyautogui
from plyer import notification
from tracking.ThreadedWebcamCapture import WebcamStream
from tracking.TrackingLogger import Logger, TrackingData, get_timestamp
from tracking.FpsMeasuring import FpsMeasurer
# from pynput import keyboard
import keyboard


# TODO:
# Warnings mit einbauen, wenn der Nutzer sich zu viel bewegt oder daten zu schlecht
# und die Zeitpunkte hier mitloggen!

# with title = pyautogui.getActiveWindow().title we could log the title of the current game if necessary


class TrackingSystem:

    def __init__(self, capturing_device):
        self.__current_frame = None
        self.__tracking_active = False

        self.frame_count = 0
        self.t1 = None
        self.t2 = None

        self.capture = capturing_device
        self.__init_logger()

        # TODO only for debugging! -> disable fps measuring later in the entire class
        self.fps_measurer = FpsMeasurer()
        capture_fps = self.capture.get_stream_fps()
        self.fps_measurer.show_optimal_fps(capture_fps)

    def __init_logger(self):
        self.__logger = Logger()
        self.__tracked_data = {key.name: None for key in TrackingData}
        self.__log_static_data()

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
        self.__logger.start_async_upload()  # init server connection and start uploading data

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

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        # save timestamp separately as it has to be the same for all the frames and the log data! otherwise it
        # can't be matched later!
        log_timestamp = get_timestamp()
        # TODO resize images to larger ones before saving?
        self.__logger.log_image("capture", frame, log_timestamp)
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

    tracking_system = TrackingSystem(capture)
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


if __name__ == "__main__":
    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Webcam eye tracking system that logs webcam images to an sftp "
                                                 "server.")
    parser.add_argument("-x", "--exec", help="Starts tracking immediately.", action="store_true")
    args = parser.parse_args()

    main()
