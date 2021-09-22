#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import os
import shutil
import sys
import time
import cv2
import pandas as pd
from post_processing.assign_load_classes import get_timestamp_from_image
from post_processing.eye_tracking.eye_tracker import EyeTracker
from post_processing.eye_tracking.image_utils import show_image_window
from post_processing.post_processing_constants import evaluation_download_folder, post_processing_log_folder
from post_processing.extract_downloaded_data import get_fps_info


def process_images(eye_tracker, participants_folders=list[str]):
    frame_count = 0
    start_time = time.time()

    # iterate over and process all images associated with a difficulty level (easy, medium and hard)
    for participant in os.listdir(evaluation_download_folder):
        if len(participants_folders) > 0 and participant not in participants_folders:
            print(f"\nSkipping folder '{participant}' as it is not in the specified folder names.\n")
            continue

        # check if there is already a post processing folder and if so, ask if it should be overwritten
        post_processing_log_path = os.path.join(evaluation_download_folder, participant, post_processing_log_folder)
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
        fps_log_path = os.path.join(evaluation_download_folder, participant, "fps_info.txt")
        fps = get_fps_info(fps_log_path)
        # set the current participant for the post processing logger
        eye_tracker.set_current_participant(participant, fps, is_evaluation_data=True)

        # iterate over the csv file with the image paths and their corresponding difficulty level
        images_label_log = os.path.join(evaluation_download_folder, participant, "labeled_images.csv")
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
                processed_frame = eye_tracker.process_current_frame(current_image, participant, difficulty_level,
                                                                    image_timestamp)

                frame_count += 1
                show_image_window(processed_frame, window_name="evaluation_processed_frame", x_pos=120, y_pos=150)
                # press q to skip to next participant / load level
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            eye_tracker.movement_tracker.save_data(participant, difficulty_level, evaluation_study_data=True)
            # after we finished one difficulty folder, log all information that was recorded for it
            eye_tracker.log_information()
            # and reset the blink detector
            eye_tracker.reset_blink_detector()

    duration = time.time() - start_time
    print(f"[INFO]: Frame Count: {frame_count}")
    print(f"[INFO]: Duration: {duration} seconds")

    # cleanup
    cv2.destroyAllWindows()
    sys.exit(0)


def start_extracting_evaluation_eye_features(participant_list=list[str], enable_annotation=False):
    eye_tracker = EyeTracker(enable_annotation, debug_active=False)
    process_images(eye_tracker, participant_list)


if __name__ == "__main__":
    # setup an argument parser to enable command line parameters
    parser = argparse.ArgumentParser(description="Eye feature postprocessing system for the evaluation data.")
    parser.add_argument("-a", "--enable_annotation", help="If enabled the tracked face parts are shown in "
                                                          "separate frames", action="store_true")
    args = parser.parse_args()
    annotation_enabled = args.enable_annotation

    # for easier debugging; select the participants that should be processed; pass empty list to process all
    participants = []
    start_extracting_evaluation_eye_features(participant_list=participants, enable_annotation=annotation_enabled)
