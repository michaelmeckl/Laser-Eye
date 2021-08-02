#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import pathlib
import sys
import shutil
import time
from py7zr import py7zr
from post_processing.post_processing_constants import download_folder, image_folder, logs_folder


def unzip(participant_folder, result_dir, file):
    if result_dir == image_folder:
        archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent / download_folder / participant_folder / "images" /
                                     file, mode="r")
    else:
        archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent / download_folder / participant_folder /
                                     "game_log.7z", mode="r")
    archive.extractall(pathlib.Path(download_folder) / participant_folder / result_dir)
    archive.close()


def extract_zipped_images(participant_folder):
    """
    extract the zipped images to a new folder and flatten the hierarchy so all images are directly in the images
    subfolder in each participant folder
    """

    print(f"\n####################\nUnzipping image data for participant {participant_folder} ...")
    for zipped_file in os.listdir(os.path.join(download_folder, participant_folder, "images")):
        unzip(participant_folder, image_folder, zipped_file)
    print("\n####################\n")

    extracted_images_path = os.path.join(download_folder, participant_folder, image_folder, "tracking_data", "images")
    result_images_dir = os.path.join(download_folder, participant_folder, image_folder)

    # flatten the hierarchy and remove obsolete folders
    start_time = time.time()
    for sub_folder in os.listdir(extracted_images_path):
        start = time.time()
        for image in os.listdir(os.path.join(extracted_images_path, sub_folder)):
            shutil.move(os.path.join(extracted_images_path, sub_folder, image),
                        os.path.join(result_images_dir, image))

        end = time.time()
        print(f"Copying folder {sub_folder} took {(end - start):.2f} seconds.")

    end_time = time.time()
    print(f"Copying participant folder {participant_folder} took {(end_time - start_time):.2f} seconds.")
    # delete the now empty unzipped folder
    shutil.rmtree(os.path.join(download_folder, participant_folder, image_folder, "tracking_data"))


def extract_game_logs(participant_folder):
    game_log_path = os.path.join(download_folder, participant_folder, "game_log.7z")
    if not os.path.exists(game_log_path):
        sys.stderr.write(f"\nGame Log zip file not found in {download_folder}/{participant_folder}!\n")
        return

    print(f"\n####################\nUnzipping game log for participant {participant_folder} ...")
    # unzip file
    unzip(participant_folder, logs_folder, game_log_path)
    print("\n####################\n")

    # and flatten the hierarchy
    extracted_logs_path = os.path.join(download_folder, participant_folder, logs_folder, "Game_Data", "StudyLogs")
    result_logs_dir = os.path.join(download_folder, participant_folder, logs_folder)
    for element in os.listdir(extracted_logs_path):
        element_path = os.path.join(extracted_logs_path, element)
        if os.path.isfile(element_path):
            shutil.move(element_path, os.path.join(result_logs_dir, element))
        else:
            new_dir_path = os.path.join(result_logs_dir, element)
            if not os.path.exists(new_dir_path):
                os.mkdir(new_dir_path)

            for file in os.listdir(element_path):
                shutil.move(os.path.join(element_path, file), os.path.join(new_dir_path, file))

    shutil.rmtree(os.path.join(download_folder, participant_folder, logs_folder, "Game_Data"))


def extract_data(participant_list=list[str]):
    for participant in os.listdir(download_folder):
        # if specific participants are given, skip the others
        if len(participant_list) > 0 and participant not in participant_list:
            print(f"\nSkipping folder {participant} as it is not in the specified participant list.\n")
            continue

        # check if there is already an extracted folder for this participant
        if "extracted_images" in os.listdir(os.path.join(download_folder, participant)):
            print(f"Participant '{participant}' already contains an extracted_images subfolder!")
            answer = input("Do you want to overwrite it? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                print(f"\nOverwriting {participant}...\n")
            else:
                print(f"\nSkipping participant '{participant}'.\n")
                continue

        # extract the .7z files
        extract_zipped_images(participant)
        extract_game_logs(participant)


if __name__ == "__main__":
    # empty list means we want to extract all participants
    extract_data(participant_list=["participant_6", "participant_7", "participant_8", "participant_9",
                                   "participant_10"])
    print("\n####################\nFinished extracting data\n####################\n")