#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import os
import shutil
import sys
import csv
import time
import cv2
from datetime import datetime
import pandas as pd
from post_processing.assign_load_classes import get_timestamp_from_image
from post_processing.eye_tracking.eye_tracker import EyeTracker
from post_processing.eye_tracking.image_utils import show_image_window
from post_processing.post_processing_constants import download_folder, post_processing_log_folder
from post_processing.extract_downloaded_data import get_fps_info


def debug_postprocess(enable_annotation, video_file_path):
    """
    Used for debugging the eye tracker functionality "live" with own webcam or video file.
    """
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


def create_eye_region_csv(participant_folder, image_folder_name):
    post_processing_path = os.path.join(download_folder, participant_folder, post_processing_log_folder)
    csv_file_path = os.path.join(download_folder, participant_folder, "labeled_eye_regions.csv")

    with open(csv_file_path, "w", newline='') as csv_file:
        # Using csv writer instead of pandas as this is far more efficient, pandas would require almost 2 minutes for
        # what can be done with the csv writer in under a second
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["image_path", "participant", "difficulty"])  # write header row first

        for difficulty_sub_dir in os.listdir(post_processing_path):
            eye_region_dir_path = os.path.join(post_processing_path, difficulty_sub_dir, image_folder_name)
            start_time = time.time()
            for idx, image_file in enumerate(os.listdir(eye_region_dir_path)):
                full_image_path = os.path.join(eye_region_dir_path, image_file)
                writer.writerow([full_image_path, participant_folder, difficulty_sub_dir])

            end_time = time.time()
            print(f"Writing eye regions to csv for difficulty {difficulty_sub_dir} took"
                  f" {(end_time - start_time):.2f} seconds.")

    print(f"Finished writing eye region csv file for {participant_folder}.")


def process_images(eye_tracker, participants_folders=list[str]):
    frame_count = 0
    start_time = time.time()

    # iterate over and process all images associated with a difficulty level (easy, medium and hard)
    for participant in os.listdir(download_folder):
        if len(participants_folders) > 0 and participant not in participants_folders:
            print(f"\nSkipping folder '{participant}' as it is not in the specified folder names.\n")
            continue

        # check if there is already a post processing folder and if so, ask if it should be overwritten
        post_processing_log_path = os.path.join(download_folder, participant, post_processing_log_folder)
        if os.path.exists(post_processing_log_path):
            print(f"A post processing folder already exists for participant '{participant}'!")
            answer = input("Do you want to overwrite it? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                print(f"\nOverwriting {participant}...\n")
                shutil.rmtree(post_processing_log_path)
            else:
                print(f"\nSkipping folder '{participant}'.\n")
                continue

        # get the recorded fps for the current participant
        fps_log_path = os.path.join(download_folder, participant, "fps_info.txt")
        fps = get_fps_info(fps_log_path)
        # set the current participant for the post processing logger
        eye_tracker.set_current_participant(participant, fps)

        # iterate over the csv file with the image paths and their corresponding difficulty level
        images_label_log = os.path.join(download_folder, participant, "labeled_images.csv")
        labeled_images_df = pd.read_csv(images_label_log)

        for difficulty_level in labeled_images_df.difficulty.unique():
            print(f"Processing images for '{participant}'; current difficulty: {difficulty_level}")
            eye_tracker.set_current_difficulty(difficulty_level)

            # create a subset of the df that contains only the rows with this difficulty level
            sub_df = labeled_images_df[labeled_images_df.difficulty == difficulty_level]
            for idx, row in sub_df.iterrows():
                image_path = row["image_path"]
                current_image = cv2.imread(image_path)

                # get the original timestamp from image so it can be associated later
                image_timestamp = get_timestamp_from_image(image_path)
                processed_frame = eye_tracker.process_current_frame(current_image, image_timestamp)

                frame_count += 1
                show_image_window(processed_frame, window_name="processed_frame", x_pos=120, y_pos=50)
                # press q to skip to next participant / load level
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # after we finished one difficulty folder, log all information that was recorded for it
            eye_tracker.log_information()
            # and reset the blink detector
            eye_tracker.reset_blink_detector()

        # after we finished one participant folder, create a csv file with the paths to the new eye region images so
        # they can be easily found by the machine learning model
        print("Creating csv file for eye regions ...")
        create_eye_region_csv(participant, image_folder_name="eye_regions")

    duration = time.time() - start_time
    print(f"[INFO]: Frame Count: {frame_count}")
    print(f"[INFO]: Duration: {duration} seconds")

    # cleanup
    # eye_tracker.stop_tracking()
    cv2.destroyAllWindows()
    sys.exit(0)


def start_extracting_eye_features(participant_list=list[str], debug=False, enable_annotation=False, video_file_path=None):
    if debug:
        debug_postprocess(enable_annotation, video_file_path)
    else:
        eye_tracker = EyeTracker(enable_annotation, debug_active=False)
        process_images(eye_tracker, participant_list)


if __name__ == "__main__":
    """
    Takes around 4 hours to run for all 18 participants.
    """

    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Postprocessing system to find the useful data in the recorded "
                                                 "images.")
    parser.add_argument("-v", "--video_file", help="path to a video file to be used instead of the webcam", type=str)
    parser.add_argument("-a", "--enable_annotation", help="If enabled the tracked face parts are highlighted in the "
                                                          "current frame", action="store_true")
    args = parser.parse_args()
    annotation_enabled = args.enable_annotation
    video_file = args.video_file

    # for easier debugging; select the participants that should be processed; pass empty list to process all
    participants = ["participant_18"]
    start_extracting_eye_features(debug=False, participant_list=participants, enable_annotation=annotation_enabled,
                                  video_file_path=video_file)
