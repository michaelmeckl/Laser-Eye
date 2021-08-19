#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import os
import shutil
import sys
import time
import cv2
from datetime import datetime
import pandas as pd
import tensorflow as tf
from machine_learning_predictor.machine_learning_constants import NEW_IMAGE_SIZE
from post_processing.eye_tracking.eye_tracker import EyeTracker
from post_processing.eye_tracking.image_utils import show_image_window
from post_processing.post_processing_constants import download_folder, labeled_images_folder, image_folder, \
    post_processing_log_folder
from post_processing.process_downloaded_data import get_fps_info


def debug_postprocess(enable_annotation, video_file_path):
    # uses the webcam or a given video file for the processing & annotation instead of the images from the participants
    if args.video_file:
        # use a custom threaded video captures to increase fps;
        # see https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
        from post_processing.eye_tracking.ThreadedFileVideoCapture import FileVideoStream
        capture = FileVideoStream(path=video_file_path, transform=None)
    else:
        # fall back to webcam (0) if no input video was provided
        capture = cv2.VideoCapture(0)

    video_width, video_height = capture.get(3), capture.get(4)
    print(f"Capture Width: {video_width}, Capture Height: {video_height}")
    eye_tracker = EyeTracker(enable_annotation, debug_active=True)

    c = 0
    start_time = datetime.now()
    while True:
        return_val, curr_frame = capture.read()
        if curr_frame is None:
            break
        c += 1

        eye_tracker.set_camera_matrix(video_width, video_height)
        processed_frame = eye_tracker.process_current_frame(curr_frame)

        # show fps in output image
        elapsed_time = (datetime.now() - start_time).total_seconds()
        fps = c / elapsed_time if elapsed_time != 0 else c
        cv2.putText(processed_frame, f"mainthread FPS: {fps:.3f}",
                    (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("fps_main_thread", processed_frame)

        # press q to quit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # eye_tracker.stop_tracking()
            cv2.destroyAllWindows()
            break


def process_images(eye_tracker, use_all_images=False, use_folder=False):
    # for easier debugging; select the participants that should be processed; pass empty list to process all
    participants = ["participant_3", "participant_10", "participant_14"]

    frame_count = 0
    start_time = time.time()
    # iterate over and process all images in the labeled images subfolders (easy, medium and hard)
    for sub_folder in os.listdir(download_folder):
        if len(participants) > 0 and sub_folder not in participants:
            print(f"\nSkipping folder '{sub_folder}' as it is not in the specified folder names.\n")
            continue

        post_processing_log_path = os.path.join(download_folder, sub_folder, post_processing_log_folder)
        if os.path.exists(post_processing_log_path):
            print(f"A post processing folder already exists for participant '{sub_folder}'!")
            answer = input("Do you want to overwrite it? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                print(f"\nOverwriting {sub_folder}...\n")
                shutil.rmtree(post_processing_log_path)
            else:
                print(f"\nSkipping folder '{sub_folder}'.\n")
                continue

        # get the recorded fps for the current participant
        fps_log_path = os.path.join(download_folder, sub_folder, "fps_info.txt")
        fps = get_fps_info(fps_log_path)
        # set the current participant for the post processing logger
        eye_tracker.set_current_participant(sub_folder, fps)

        """
        if use_all_images:
            # process all extracted images, even the ones that aren't useful for the machine learning model
            images_path = os.path.join(download_folder, sub_folder, image_folder)
            for image_file in os.listdir(images_path):
                current_frame = cv2.imread(os.path.join(images_path, image_file))
                processed_frame = eye_tracker.process_current_frame(current_frame)

                frame_count += 1
                show_image_window(processed_frame, window_name="processed_frame", x_pos=120, y_pos=50)
                # press q to skip to next participant / load level
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        """
        # only process the labeled images, that can be associated with one of the load levels
        if use_folder:
            images_path = os.path.join(download_folder, sub_folder, labeled_images_folder)
            for load_folder in os.listdir(images_path):
                print(f"Processing images for '{sub_folder}'; current difficulty: {load_folder}")
                eye_tracker.set_current_difficulty(load_folder)

                for image_file in os.listdir(os.path.join(images_path, load_folder)):
                    current_frame = cv2.imread(os.path.join(images_path, load_folder, image_file))
                    processed_frame = eye_tracker.process_current_frame(current_frame)

                    frame_count += 1
                    show_image_window(processed_frame, window_name="processed_frame", x_pos=120, y_pos=50)
                    # press q to skip to next participant / load level
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # after we finished one load folder, log all information that was recorded for it
                eye_tracker.log_information()
                # and reset the blink detector
                eye_tracker.reset_blink_detector()
        else:
            # iterate over the csv file and yield the image paths and their corresponding difficulty level
            images_label_log = os.path.join(download_folder, sub_folder, "labeled_images.csv")
            labeled_images_df = pd.read_csv(images_label_log)

            for difficulty_level in labeled_images_df.difficulty.unique():
                print(f"Processing images for '{sub_folder}'; current difficulty: {difficulty_level}")
                eye_tracker.set_current_difficulty(difficulty_level)

                # create a subset of the df that contains only the rows with this difficulty level
                sub_df = labeled_images_df[labeled_images_df.difficulty == difficulty_level]
                for idx, row in sub_df.iterrows():
                    image_path = row["image_path"]
                    current_image = cv2.imread(image_path)

                    # TODO resizing needs to be done later in the process as it causes problems in correctly finding
                    #  faces and pupils for some participants
                    """
                    # crop or pad image depending on it's size
                    current_image = tf.image.resize_with_crop_or_pad(current_image,
                                                                   target_height=NEW_IMAGE_SIZE[1],
                                                                   target_width=NEW_IMAGE_SIZE[0]).numpy()
                    """
                    processed_frame = eye_tracker.process_current_frame(current_image)

                    frame_count += 1
                    show_image_window(processed_frame, window_name="processed_frame", x_pos=120, y_pos=50)
                    # press q to skip to next participant / load level
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # after we finished one load folder, log all information that was recorded for it
                eye_tracker.log_information()
                # and reset the blink detector
                eye_tracker.reset_blink_detector()

    duration = time.time() - start_time
    print(f"[INFO]: Frame Count: {frame_count}")
    print(f"[INFO]: Duration: {duration} seconds")

    # cleanup
    # eye_tracker.stop_tracking()
    cv2.destroyAllWindows()
    sys.exit(0)


def start_extracting_features(debug=False, enable_annotation=False, video_file_path=None):
    if debug:
        debug_postprocess(enable_annotation, video_file_path)
    else:
        eye_tracker = EyeTracker(enable_annotation, debug_active=False)
        process_images(eye_tracker)


if __name__ == "__main__":
    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Postprocessing system to find the useful data in the recorded "
                                                 "images.")
    parser.add_argument("-v", "--video_file", help="path to a video file to be used instead of the webcam", type=str)
    parser.add_argument("-a", "--enable_annotation", help="If enabled the tracked face parts are highlighted in the "
                                                          "current frame", action="store_true")
    args = parser.parse_args()
    annotation_enabled = args.enable_annotation
    video_file = args.video_file

    start_extracting_features(debug=False, enable_annotation=annotation_enabled, video_file_path=video_file)
