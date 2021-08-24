#!/usr/bin/python3
# -*- coding:utf-8 -*-

import itertools
import os
import pathlib
import random
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import RANDOM_SEED, data_folder_path
from machine_learning_predictor.ml_utils import show_sample_images, set_random_seed
from post_processing.post_processing_constants import download_folder, labeled_images_folder
from post_processing.process_downloaded_data import get_fps_info


# TODO two different sizes for width and height?
NEW_IMAGE_SIZE = 128


def get_participant_images(participant_folder, use_folder=False, use_step_size=False):
    post_processing_folder_path = pathlib.Path(__file__).parent.parent / "post_processing"
    participant_folder_path = post_processing_folder_path / download_folder / participant_folder
    subset_size = 150

    if use_step_size:
        subset_size = 2000
        # TODO step_size good or bad idea ?
        fps_log_path = os.path.join(data_folder_path, participant_folder, "fps_info.txt")
        fps = get_fps_info(fps_log_path)
        # print("FPS:", fps)
        step_size = int(round(fps))  # take only one image per second ?

    if not use_folder:
        # iterate over the csv file and yield the image paths and their corresponding difficulty level
        # images_label_log = participant_folder_path / "labeled_images.csv"
        images_label_log = participant_folder_path / "labeled_eye_regions.csv"
        labeled_images_df = pd.read_csv(images_label_log)

        """
        # less flexible version:
        for idx, row in labeled_images_df.iterrows():
            difficulty_level = row["difficulty"]
            image_path = row["image_path"]
            full_image_path = os.path.join(post_processing_folder_path, image_path)
            # current_image = cv2.imread(full_image_path)
            yield full_image_path, difficulty_level
        """

        # FIXME unfortunately a different order changes the results :(
        # for difficulty_level in ["easy", "hard", "medium"]:
        for difficulty_level in labeled_images_df.difficulty.unique():
            # create a subset of the df that contains only the rows with this difficulty level
            sub_df = labeled_images_df[labeled_images_df.difficulty == difficulty_level]

            # TODO take only small subset for faster testing
            if use_step_size:
                df_iterator = list(itertools.islice(sub_df.iterrows(), 0, subset_size, step_size))
            else:
                df_iterator = list(itertools.islice(sub_df.iterrows(), 0, subset_size))

            for idx, row in df_iterator:
                image_path = row["image_path"]
                full_image_path = os.path.join(post_processing_folder_path, image_path)
                # current_image = cv2.imread(full_image_path)
                yield full_image_path, difficulty_level

    else:
        # iterate over the labeled images folder and yield the image paths and their corresponding difficulty level
        labeled_images_path = participant_folder_path / labeled_images_folder
        for difficulty_level in os.listdir(labeled_images_path):
            # TODO only subset for faster testing
            if use_step_size:
                image_list = os.listdir(os.path.join(labeled_images_path, difficulty_level))[:subset_size:step_size]
            else:
                image_list = os.listdir(os.path.join(labeled_images_path, difficulty_level))[:subset_size]

            for image_file in image_list:
                full_image_path = os.path.join(labeled_images_path, difficulty_level, image_file)
                # current_image = cv2.imread(full_image_path)
                yield full_image_path, difficulty_level


def preprocess_train_test_data(data):
    train_test_data = []

    for participant in data:
        for image_path, difficulty_level in get_participant_images(participant, use_folder=False, use_step_size=False):
            label_vector = DifficultyLevels.get_one_hot_encoding(difficulty_level)
            try:
                # grayscale_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # use this instead: (for reason see
                # https://stackoverflow.com/questions/37203970/opencv-grayscale-mode-vs-gray-color-conversion#comment103382641_37208336)
                color_img = cv2.imread(image_path)
                grayscale_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                cv2.imshow("grayscale", grayscale_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                resized_img = cv2.resize(grayscale_img, (NEW_IMAGE_SIZE, NEW_IMAGE_SIZE))
                train_test_data.append([np.array(resized_img, dtype=np.uint8), label_vector, image_path])

            except Exception as e:
                sys.stderr.write(f"\nError in reading and resizing image '{image_path}': {e}")

    random.shuffle(train_test_data)
    return train_test_data


def start_preprocessing():
    set_random_seed(RANDOM_SEED)  # for reproducibility

    without_participants = ["participant_1", "participant_5"]  # "participant_6"
    # without_participants = []

    all_participants = os.listdir(data_folder_path)[:12]  # TODO only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    # all_participants = list(set(all_participants) - set(without_participants))  # doesn't keep the original order!
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    random.shuffle(all_participants)

    # TODO random choice one to use as test set ?

    train_ratio = 0.8
    train_split = int(len(all_participants) * train_ratio)
    train_participants = all_participants[:train_split]
    test_participants = all_participants[train_split:]
    print(f"{len(train_participants)} participants used for training: {train_participants}")
    print(f"{len(test_participants)} participants used for validation: {test_participants}")

    train_data = preprocess_train_test_data(train_participants)
    test_data = preprocess_train_test_data(test_participants)
    print("Len training data: ", len(train_data))
    print("Len test data: ", len(test_data))

    # TODO save them all as one and split later when reading in?
    train_images = []  # features for training
    train_labels = []  # labels for training
    train_paths = []  # paths to images in train data
    for img_data, label, path in train_data:
        train_images.append(img_data)
        train_labels.append(label)
        train_paths.append(path)

    test_images = []  # features for testing
    test_labels = []  # labels for testing
    test_paths = []  # paths to images in test data
    for img_data, label, path in test_data:
        test_images.append(img_data)
        test_labels.append(label)
        test_paths.append(path)

    show_sample_images([data[0] for data in train_data[25:]], [data[1] for data in train_data[25:]])

    train_images = np.asarray(train_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)
    test_images = np.asarray(test_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)
    # normalize all images to [0, 1] for the neural network
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()

    result_folder = "ml_results"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # TODO use np.savez() to save compressed?
    np.save(os.path.join(result_folder, "train_images.npy"), train_images, allow_pickle=False)
    np.save(os.path.join(result_folder, "train_labels.npy"), train_labels, allow_pickle=False)
    np.save(os.path.join(result_folder, "train_paths.npy"), train_paths, allow_pickle=False)

    np.save(os.path.join(result_folder, "test_images.npy"), test_images, allow_pickle=False)
    np.save(os.path.join(result_folder, "test_labels.npy"), test_labels, allow_pickle=False)
    np.save(os.path.join(result_folder, "test_paths.npy"), test_paths, allow_pickle=False)


if __name__ == "__main__":
    start_preprocessing()
