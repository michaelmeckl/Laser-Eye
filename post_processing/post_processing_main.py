#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import os
import pathlib
import sys
import shutil
import time
import cv2
from datetime import datetime
import pysftp
from paramiko.ssh_exception import SSHException
from py7zr import py7zr
from post_processing.eye_tracker import EyeTracker
from tracking.TrackingLogger import get_server_credentials

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


download_folder = "./tracking_images_download"
image_folder = "./extracted_images"


def debug_postprocess(enable_annotation, show_video):
    # uses the webcam or a given video file for the processing & annotation instead of the images from the participants
    if args.video_file:
        # use a custom threaded video captures to increase fps;
        # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
        from post_processing.ThreadedFileVideoCapture import FileVideoStream
        capture = FileVideoStream(path=args.video_file, transform=None)
    else:
        # fall back to webcam (0) if no input video was provided
        capture = cv2.VideoCapture(0)

    video_width, video_height = capture.get(3), capture.get(4)
    print(f"Capture Width: {video_width}, Capture Height: {video_height}")
    eye_tracker = EyeTracker(video_width, video_height, enable_annotation, show_video)

    c = 0
    start_time = datetime.now()
    while True:
        return_val, curr_frame = capture.read()
        if curr_frame is None:
            break
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


def download_images_from_server():
    credentials = get_server_credentials(pathlib.Path(__file__).parent.parent / "sftp_credentials.properties")
    if credentials is None:
        sys.stderr.write("Reading sftp server credentials didn't work! Terminating program...")
        sys.exit(1)

    hostname = credentials["sftp_hostname"]
    username = credentials["sftp_username"]
    password = credentials["sftp_password"]
    port = credentials.getint("sftp_port")
    # NOTE: setting hostkeys to None is highly discouraged as we lose any protection
    # against man-in-the-middle attacks; however, it is the easiest solution and for now it should be fine;
    # see https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp for correct solution
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    try:
        sftp = pysftp.Connection(host=hostname, username=username, password=password, port=port, cnopts=cnopts)
    except SSHException as e:
        sys.stderr.write(f"Die Verbindung zum Server konnte nicht hergestellt werden! Bitte stellen Sie sicher, dass "
                         f"sie während des Trackings eine stabile Internetverbindung haben! Fehler: {e}")
        sys.exit(1)

    # create the local folder to put the downloaded files into
    if not os.path.exists(download_folder):
        os.mkdir(download_folder)

    # iterate over all folders on the server and download all zipped image files per participant
    root_dir = "home"
    for i, directory in enumerate(sftp.listdir(root_dir)):
        # important to not use os.path.join as this would use the local machine's path separator which might not be
        # the same as on the server!
        images_path = f"{root_dir}/{directory}/images"
        if sftp.isdir(images_path):
            print(f"Downloading folder {i}:  {images_path}")
            local_folder = os.path.join(download_folder, f"participant_{i}")
            if not os.path.exists(local_folder):
                os.mkdir(local_folder)
            sftp.get_d(images_path, local_folder)
        else:
            sys.stderr.write(f"\nNo images subfolder found in {root_dir}/{directory}! Skipping this folder.")


def unzip(participant_folder, file):
    archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent / download_folder / participant_folder / file, mode="r")
    archive.extractall(pathlib.Path(image_folder) / participant_folder)
    archive.close()


def extract_zipped_images():
    """
    extract the zipped images to a new folder and flatten the hierarchy so all images are in each participant folder
    """

    # unzip .7z files
    [unzip(participant, zipped_file) for participant in os.listdir(download_folder)
     for zipped_file in os.listdir(os.path.join(download_folder, participant))]

    for participant_folder in os.listdir(image_folder):
        extracted_images_path = os.path.join(image_folder, participant_folder, "tracking_data", "images")
        result_images_dir = os.path.join(image_folder, participant_folder)

        start_11 = time.time()
        for sub_folder in os.listdir(extracted_images_path):
            start = time.time()
            for image in os.listdir(os.path.join(extracted_images_path, sub_folder)):
                shutil.move(os.path.join(extracted_images_path, sub_folder, image),
                            os.path.join(result_images_dir, image))

            end = time.time()
            print(f"Copying folder {sub_folder} took {(end - start):.2f} seconds.")

        end_11 = time.time()
        print(f"Copying participant folder {participant_folder} took {(end_11 - start_11):.2f} seconds.")
        shutil.rmtree(os.path.join(image_folder, participant_folder, "tracking_data"))  # delete temporary folder


def process_images(eye_tracker):
    frame_count = 0
    start_time = time.time()
    for sub_folder in os.listdir(image_folder):
        for image_file in os.listdir(os.path.join(image_folder, sub_folder)):
            current_frame = cv2.imread(os.path.join(image_folder, sub_folder, image_file))
            processed_frame = eye_tracker.process_current_frame(current_frame)

            frame_count += 1
            cv2.imshow("processed_frame", processed_frame)
            # press q to quit earlier
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    duration = time.time() - start_time
    print(f"[INFO]: Frame Count: {frame_count}")
    print(f"[INFO]: Duration: {duration} seconds")
    # print(f"[INFO]: FPS: {duration / frame_count:.3f}")

    # cleanup
    cv2.destroyAllWindows()
    sys.exit(0)


def main(debug: bool):
    enable_annotation = args.enable_annotation
    show_video = args.show_video

    if debug:
        debug_postprocess(enable_annotation, show_video)

    else:
        frame_width, frame_height = None, None

        # TODO downloading takes for ever; enable at own risk :)
        # download_images_from_server()  # download the image data from our sftp server

        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        extract_zipped_images()

        # get size of first image; TODO dirty way for now, improve later
        for sub_folder in os.listdir(image_folder):
            for image in os.listdir(os.path.join(image_folder, sub_folder)):
                # this is the first image
                first_image = cv2.imread(os.path.join(image_folder, sub_folder, image))
                frame_width = first_image.shape[1]
                frame_height = first_image.shape[0]
                break  # we only want the first image, so we stop immediately

        if frame_width is None:
            print("first image doesn't seem to exist!")
            return

        # TODO resize all images to the same size for head pose estimator!! e.g. use keras.flow_from_directory ??
        eye_tracker = EyeTracker(frame_width, frame_height, enable_annotation, show_video)
        process_images(eye_tracker)


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

    main(debug=False)

    """
    archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent.parent / f"1.7z", mode="r")
    archive.extractall(path=pathlib.Path(__file__).parent.parent)
    archive.close()
    """
