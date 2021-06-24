#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import cv2
from datetime import datetime
from post_processing.eye_tracker import EyeTracker


# we need:
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


def main():
    enable_annotation = args.enable_annotation
    show_video = args.show_video

    # use a custom threaded video captures to increase fps;
    # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    if args.video_file:
        from post_processing.ThreadedFileVideoCapture import FileVideoStream
        capture = FileVideoStream(path=args.video_file, transform=None)
    else:
        from tracking.ThreadedWebcamCapture import WebcamStream
        # fall back to webcam (0) if no input video was provided
        capture = WebcamStream(src=0)

    video_width, video_height = capture.get_stream_dimensions()
    print(f"Capture Width: {video_width}, Capture Height: {video_height}")

    eye_tracker = EyeTracker(video_width, video_height, enable_annotation, show_video)

    c = 0
    start_time = datetime.now()

    while True:
        curr_frame = capture.read()
        if curr_frame is None:
            continue
        c += 1

        processed_frame = eye_tracker.process_current_frame(curr_frame)

        # show fps in output image
        elapsed_time = (datetime.now() - start_time).total_seconds()
        fps = c / elapsed_time if elapsed_time != 0 else c
        cv2.putText(processed_frame, f"mainthread FPS: {fps:.3f}",
                    (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("fps_main_thread", processed_frame)

        # press q to quit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Postprocessing system to find the useful data in the recorded "
                                                 "images.")
    parser.add_argument("-v", "--video_file", help="path to a video file to be used instead of the webcam", type=str)
    parser.add_argument("-a", "--enable_annotation", help="If enabled the tracked face parts are highlighted in the "
                                                          "current frame", action="store_true")
    parser.add_argument("-s", "--show_video", help="If enabled the given video or the webcam recoding is shown in a "
                                                   "separate window", action="store_true")
    args = parser.parse_args()

    main()
