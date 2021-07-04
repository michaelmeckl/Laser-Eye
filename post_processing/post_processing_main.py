#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import os
import sys
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


def main(debug=False):
    enable_annotation = args.enable_annotation
    show_video = args.show_video

    if debug:
        # use a custom threaded video captures to increase fps;
        # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
        if args.video_file:
            from post_processing.ThreadedFileVideoCapture import FileVideoStream
            capture = FileVideoStream(path=args.video_file, transform=None)
        else:
            from post_processing.ThreadedWebcamCapture import WebcamStream
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
                cv2.destroyAllWindows()
                break
    else:
        image_folder = "./tracking_data/images"
        frame_width, frame_height = None, None

        # TODO unzip .7z files first!
        """
        archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent / f"{filename}.7z", mode="r")
        archive.extractall(path=pathlib.Path(__file__).parent.parent)
        archive.close()
        """

        for sub_folder in os.listdir(image_folder):
            for image in os.listdir(f"{image_folder}/{sub_folder}"):
                first_image = cv2.imread(f"{image_folder}/{sub_folder}/{image}")
                frame_width = first_image.shape[1]
                frame_height = first_image.shape[0]
                break  # we only want the first image

        if frame_width is None:
            print("first image doesn't seem to exist!")
            return

        # TODO resize all images to the same size for head pose estimator?? e.g. use keras.flow_from_directory ??
        eye_tracker = EyeTracker(frame_width, frame_height, enable_annotation, show_video)

        for sub_folder in os.listdir(image_folder):
            for image_file in os.listdir(f"{image_folder}/{sub_folder}"):
                current_frame = cv2.imread(f"{image_folder}/{sub_folder}/{image_file}")
                processed_frame = eye_tracker.process_current_frame(current_frame)

                cv2.imshow("processed_frame", processed_frame)
                # press q to quit earlier
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # cleanup
        cv2.destroyAllWindows()
        sys.exit(0)


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
