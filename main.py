#!/usr/bin/python3
# -*- coding:utf-8 -*-

from turtle import Turtle, Screen
import argparse
import datetime
import sys
import threading
import cv2
import pyautogui
from eye_tracker import EyeTracker
from utils.FpsMeasuring import FPS
from utils.ThreadedFileVideoCapture import FileVideoStream
from utils.ThreadedWebcamCapture import WebcamStream


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


def cleanup_image_capture(capture):
    """
    Stop and cleanup active video/webcam captures and destroy open windows if any.
    """
    # capture.release()
    capture.stop()
    cv2.destroyAllWindows()


class TrackingSystem:
    def __init__(self):
        self._current_frame = None
        self.gaze_point_l, self.gaze_point_r = None, None
        self.stop_tracking = False
        self.stop_gaze_tracking = False

    def get_current_frame(self):
        return self._current_frame

    def run_gaze_tracking(self):
        self.__screenWidth, self.__screenHeight = pyautogui.size()  # Get the size of the primary monitor.
        screen = Screen()
        screen.setup(self.__screenWidth, self.__screenHeight)
        screen.setworldcoordinates(0, 0, self.__screenWidth, self.__screenHeight)
        self.__turt = Turtle(visible=True)
        self.__turt.speed('fastest')

        while True:
            if self.stop_gaze_tracking:
                break

            if self.gaze_point_l is None or self.gaze_point_r is None:
                continue

            self.__turt.penup()

            radius = 5
            self.__turt.setposition(self.gaze_point_l[0], self.gaze_point_l[1])
            self.__turt.dot(radius * 2, "green")

            self.__turt.setposition(self.gaze_point_r[0], self.gaze_point_r[1])
            self.__turt.dot(radius * 2, "blue")

            x = (self.gaze_point_l[0] + self.gaze_point_r[0]) / 2
            y = (self.gaze_point_l[1] + self.gaze_point_r[1]) / 2
            avg_point = (x, y)  # - as turtle draws inverse in y direction
            self.__turt.setposition(avg_point)
            self.__turt.dot(8 * 2, "orange")

    def stop(self):
        self.stop_tracking = False

    def start_tracking(self, capture, eye_tracker):
        c = 0
        start = datetime.datetime.now()
        try:
            while True:
                if self.stop_tracking:
                    break

                frame = capture.read()
                if frame is None:
                    sys.stderr.write("Frame from stream thread is None! This shouldn't happen!")
                    sys.exit(1)

                processed_frame = eye_tracker.process_current_frame(frame)
                # out.write(processed_frame)  # write current frame to video

                self.gaze_point_l, self.gaze_point_r = eye_tracker.get_gaze_points()

                # update the FPS counter
                c += 1

                cv2.putText(processed_frame, f"FPS: {c / (datetime.datetime.now()-start).total_seconds()}",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('fps', processed_frame)

                self._current_frame = processed_frame

                # read until the video is finished or if no video was provided until the
                # user presses 'q' on the keyboard;
                # replace 1 with 0 to step manually through the video "frame-by-frame"
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # out.release()  # release output video
                    eye_tracker.stop_tracking()
                    cleanup_image_capture(capture)

                    self.stop_gaze_tracking = True
                    break

        except KeyboardInterrupt:
            # TODO should later be replaced with the keyboard library probably
            print("Press Ctrl-C to terminate while statement")
            eye_tracker.stop_tracking()
            cleanup_image_capture(capture)


def main():
    debug_active = args.debug
    enable_annotation = args.enable_annotation
    show_video = args.show_video

    # use a custom threaded video captures to increase fps;
    # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    if args.video_file:
        capture = FileVideoStream(path=args.video_file, transform=None).start()
    else:
        # fall back to webcam (0) if no input video was provided
        capture = WebcamStream(src=0).start()

    video_width, video_height = capture.get_stream_dimensions()
    cap_fps = capture.get_stream_fps()
    fps_sleep = int(1000 / cap_fps)
    print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')
    # fps = FPS().start()  # start measuring FPS  # TODO only for debugging!! -> disable later

    eye_tracker = EyeTracker(video_width, video_height, debug_active, enable_annotation, show_video)

    """
    # record video
    fourcc2 = cv2.VideoWriter_fourcc(*'MP4V')  # for mp4  # H264 seems to work too
    out = cv2.VideoWriter('output.mp4', fourcc2, 10.0, (640, 480), isColor=True)  # 10 FPS
    """

    tracker = TrackingSystem()
    try:
        t = threading.Thread(target=tracker.start_tracking, args=(capture, eye_tracker), daemon=True)
        t.start()

        # tracker.run_gaze_tracking()

        """
        _start = datetime.datetime.now()
        while True:
            f = tracker.get_current_frame()
            if f is None:
                continue

            fps.update()
            # if f.size:
            #    cv2.imwrite(f'test_images/frame__{get_timestamp()}.png', f)

            if cv2.waitKey(1) & 0xFF == ord('e'):
                tracker.stop()
                fps.stop()
                print(f"[INFO] elasped time: {fps.elapsed():.2f}")
                print(f"[INFO] approx. FPS: {fps.fps():.2f}")
                break
        """

    except KeyboardInterrupt:
        # TODO should later be replaced with the keyboard library probably
        print("Press Ctrl-C to terminate while statement")
        eye_tracker.stop_tracking()
        cleanup_image_capture(capture)

        tracker.stop()

        # # stop the timer and display FPS information
        # fps.stop()
        # print(f"[INFO] elasped time: {fps.elapsed():.2f}")
        # print(f"[INFO] approx. FPS: {fps.fps():.2f}")


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
    args = parser.parse_args()

    main()
