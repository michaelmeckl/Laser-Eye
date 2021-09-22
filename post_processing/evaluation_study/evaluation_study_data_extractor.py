#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import pathlib
import shutil
import time
from py7zr import py7zr
from post_processing.post_processing_constants import evaluation_download_folder, image_folder


def unzip(participant_folder, result_dir, file):
    archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent / evaluation_download_folder / participant_folder /
                                 "images" / file, mode="r")
    archive.extractall(pathlib.Path(evaluation_download_folder) / participant_folder / result_dir)
    archive.close()


def extract_zipped_images(participant_folder):
    print(f"\n####################\nUnzipping image data for {participant_folder} ...")
    for zipped_file in os.listdir(os.path.join(evaluation_download_folder, participant_folder, "images")):
        unzip(participant_folder, image_folder, zipped_file)
    print(f"Finished unzipping image data for {participant_folder}\n")

    extracted_images_path = os.path.join(evaluation_download_folder, participant_folder, image_folder, "tracking_data", "images")
    result_images_dir = os.path.join(evaluation_download_folder, participant_folder, image_folder)

    # flatten the hierarchy and remove obsolete folders
    start_time = time.time()
    for sub_folder in os.listdir(extracted_images_path):
        for image in os.listdir(os.path.join(extracted_images_path, sub_folder)):
            shutil.move(os.path.join(extracted_images_path, sub_folder, image),
                        os.path.join(result_images_dir, image))

    end_time = time.time()
    # print(f"Copying participant folder {participant_folder} took {(end_time - start_time):.2f} seconds.")

    # delete the now empty unzipped folder
    shutil.rmtree(os.path.join(evaluation_download_folder, participant_folder, image_folder, "tracking_data"))


def extract_evaluation_data(participant_list=list[str]):
    for participant in os.listdir(evaluation_download_folder):
        # if specific participants are given, skip the others
        if len(participant_list) > 0 and participant not in participant_list:
            print(f"\nSkipping folder {participant} as it is not in the specified participant list.\n")
            continue

        # check if there is already an extracted folder for this participant
        if "extracted_images" in os.listdir(os.path.join(evaluation_download_folder, participant)):
            print(f"Participant '{participant}' already contains an extracted_images subfolder!")
            answer = input("Do you want to overwrite it? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                print(f"\nOverwriting {participant}...\n")
            else:
                print(f"\nSkipping participant '{participant}'.\n")
                continue

        # extract the .7z files
        extract_zipped_images(participant)


if __name__ == "__main__":
    # empty list means we want to extract all participants
    extract_evaluation_data(participant_list=[])
    print("\n####################\nFinished extracting evaluation study data\n####################\n")
