#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import datetime
import os
import sys
import threading
import cv2
import pyautogui
from plyer import notification

from eye_tracker import EyeTracker
from utils.FpsMeasuring import FpsMeasurer
# from pynput import keyboard
import keyboard


# TODO we need:
"""
- webcam images of eyes and pupils ✔
- pupil positions  ✔
- pupil sizes (diameter)
- average pupilsize; peak pupil size
- fixations and saccades (count, mean, std)   ❌ # TODO
- blinks (rate, number, etc.)   ❌ (basic approaches are there; need to be expanded to actually be useful)
"""


# TODO Lösungsansätze für Problem mit unterschiedlichen Bilddimensionen pro Frame:
# 1. kleinere bilder mit padding versehen bis alle gleich groß wie größtes
# 2. größere bilder runterskalieren bis alle gleich groß wie kleinstes (oder alternativ crop)
# 3. jetzt erstmal unterschiedlich lassen und dann später beim CNN vorverarbeiten!
#      -> vermtl. eh am besten weil später neue Bilder ja auch erstmal vorverarbeitet werden müssen!
# TODO: => fürs Erste jetzt ignorieren und nur Rechtecke nehmen!

# TODO:
# add some parts to the readme on how this was updated compared to the original

# remove all parts of the tracking that isn't needed (e.g. gaze tracking probably)

# Warnings mit einbauen, wenn der Nutzer sich zu viel beweget oder daten zu schlecht und die Zeitpunkte hier mitloggen!

# doch ganzes video mit aufzeichnen? vermtl. nicht sinnvoll wegen Größe und FPS-Einbußen?

# TODO log things like the webcam fps and the screen size!!!
# maybe even ask for things like OS, CPU, RAM, GPU, etc. in a questionnaire
# -> psutil.cpu_count(logical=True) für number of cores

# with title = pyautogui.getActiveWindow().title we could log the title of the current game if necessary


class TrackingSystem:

    def __init__(self, capturing_device, eye_tracker):
        self.__current_frame = None
        self.__tracking_active = False
        # Get the size of the primary monitor.
        self.__screenWidth, self.__screenHeight = pyautogui.size()

        self.eye_tracker = eye_tracker
        self.capture = capturing_device

        # TODO only for debugging! -> disable fps measuring later in the entire class
        self.fps_measurer = FpsMeasurer()
        capture_fps = self.capture.get_stream_fps()
        self.fps_measurer.show_optimal_fps(capture_fps)

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
        print(f"Pressed required hotkey!")
        # toggle tracking on hotkey press
        if self.__tracking_active:
            self.__tracking_active = False
            self.__cleanup()
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

        # self.capture.release()
        # self.out.release()
        self.capture.stop()
        cv2.destroyAllWindows()
        self.eye_tracker.stop_tracking()

        self.fps_measurer.stop()
        print(f"[INFO] elapsed time: {self.fps_measurer.elapsed():.2f} seconds")
        print(f"[INFO] approx. FPS on background thread: {self.fps_measurer.fps():.2f}")
        print("Frames on main thread:", self.fps_measurer._numFrames)

    def __cleanup(self):
        # TODO pause tracking, dont stop it!
        # -> only pause both main capturing loops and the logging / uploading; don't kill threads!
        # self.eye_tracker.stop_tracking()
        print("paused tracking")

    def __start_tracking(self):
        """
        This function runs on a background thread.
        """
        # record video
        # fourcc2 = cv2.VideoWriter_fourcc(*'MP4V')  # for mp4  # H264 seems to work too
        # self.out = cv2.VideoWriter('output.mp4', fourcc2, 17.0, (640, 480), isColor=True)  # 17 FPS
        # self.out = cv2.VideoWriter('output.mp4', fourcc2, 12.0, (150, 150), isColor=True)

        self.fps_measurer.start()
        while True:
            if not self.__tracking_active:
                break

            frame = self.capture.read()
            if frame is None:
                # TODO does happen if video input has finished
                sys.stderr.write("Frame from stream thread is None! This shouldn't happen!")
                break

            # update the FPS counter
            self.fps_measurer.update()

            processed_frame = self.eye_tracker.process_current_frame(frame)
            self.__current_frame = processed_frame
            # if processed_frame is not None:
            #    self.out.write(processed_frame)  # write current frame to video

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
        # self.__cleanup()

    def get_current_frame(self):
        return self.__current_frame

    def debug_start_tracking(self):
        # TODO only for faster debugging; remove later
        # start tracking on a background thread
        self.__tracking_active = True
        self.capture.start()  # start reading frames from webcam
        self.tracking_thread = threading.Thread(target=self.__start_tracking, daemon=True)
        self.tracking_thread.start()


def main():
    debug_active = args.debug
    enable_annotation = args.enable_annotation
    show_video = args.show_video

    # use a custom threaded video captures to increase fps;
    # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    if args.video_file:
        from utils.ThreadedFileVideoCapture import FileVideoStream
        # TODO test if using transform in this thread would speed up things !
        capture = FileVideoStream(path=args.video_file, transform=None)
    else:
        from utils.ThreadedWebcamCapture import WebcamStream
        # fall back to webcam (0) if no input video was provided
        capture = WebcamStream(src=0)

    video_width, video_height = capture.get_stream_dimensions()
    print(f"Capture Width: {video_width}, Capture Height: {video_height}")

    eye_tracker = EyeTracker(video_width, video_height, debug_active, enable_annotation, show_video)
    tracking_system = TrackingSystem(capture, eye_tracker)
    tracking_system.listen_for_hotkey()

    print("Press ctrl + shift + a to toggle tracking and ctrl + shift + q to stop it!")
    if args.exec:
        tracking_system.debug_start_tracking()

    # TODO only for debugging:
    c = 0
    start_time = datetime.datetime.now()
    while True:
        curr_frame = tracking_system.get_current_frame()
        if curr_frame is None:
            continue
        c += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = c / elapsed_time if elapsed_time != 0 else c
        cv2.putText(curr_frame, f"mainthread FPS: {fps:.3f}",
                    (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("fps_main_thread", curr_frame)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break


if __name__ == "__main__":
    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Webcam eye tracking system that logs different facial information. "
                                                 "Press 'q' on the keyboard to stop the tracking if reading from "
                                                 "webcam.")
    parser.add_argument("-v", "--video_file", help="path to a video file to be used instead of the webcam", type=str)
    parser.add_argument("-d", "--debug", help="Enable debug mode: logged data is written to stdout instead of files",
                        action="store_true")
    parser.add_argument("-a", "--enable_annotation", help="If enabled the tracked face parts are highlighted in the "
                                                          "current frame",
                        action="store_true")
    parser.add_argument("-s", "--show_video", help="If enabled the given video or the webcam recoding is shown in a "
                                                   "separate window",
                        action="store_true")
    parser.add_argument("-x", "--exec", help="Starts tracking immediately.", action="store_true")
    args = parser.parse_args()

    main()
