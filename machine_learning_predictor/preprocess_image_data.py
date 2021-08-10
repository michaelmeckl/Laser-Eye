#!/usr/bin/python3
# -*- coding:utf-8 -*-

import itertools
import os
import pathlib
import random
import sys
from enum import Enum
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from post_processing.post_processing_constants import download_folder, labeled_images_folder, image_folder


# TODO Lösungsansätze für Problem mit unterschiedlichen Bilddimensionen pro Frame:
# 1. kleinere bilder mit padding versehen bis alle gleich groß wie größtes
# 2. größere bilder runterskalieren bis alle gleich groß wie kleinstes (oder alternativ crop)
# 3. einfach irgendeine größe nehmen und verzerrung akzeptieren:
NEW_IMAGE_SIZE = 80

data_folder_path = os.path.join(os.path.dirname(__file__), "..", "post_processing", "tracking_data_download")

random.seed(42)  # for reproducibility


def show_samples(dataset):
    fig = plt.figure(figsize=(14, 14))
    columns = 3
    rows = 3

    print(columns * rows, "samples from the dataset")
    i = 1
    for a, b in dataset.take(columns * rows):
        fig.add_subplot(rows, columns, i)
        plt.imshow(a)
        # plt.imshow(a.numpy())
        plt.title("image shape:" + str(a.shape) + " Label:" + str(b.numpy()))

        i = i + 1
    plt.show()


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        label = DifficultyLevels.get_label_for_encoding(label_batch[n])
        plt.title(label)
        plt.axis('off')
    plt.show()


def get_participant_images(participant_folder, use_folder=False):
    participant_folder_path = pathlib.Path(__file__).parent.parent / "post_processing" / download_folder / \
                              participant_folder
    subset_size = 150

    if not use_folder:
        # iterate over the csv file and yield the image paths and their corresponding difficulty level
        images_label_log = participant_folder_path / "labeled_images.csv"
        labeled_images_df = pd.read_csv(images_label_log)

        """
        # less flexible version:
        for idx, row in labeled_images_df.iterrows():
            difficulty_level = row["load_level"]
            image_name = row["image_path"]
            full_image_path = os.path.join(participant_folder_path, image_folder, image_name)
            # current_image = cv2.imread(full_image_path)
            yield full_image_path, difficulty_level
        """
        for difficulty_level in labeled_images_df.load_level.unique():
            # create a subset of the df that contains only the rows with this difficulty level
            sub_df = labeled_images_df[labeled_images_df.load_level == difficulty_level]

            df_iterator = list(itertools.islice(sub_df.iterrows(), subset_size))  # TODO only subset for faster testing
            for idx, row in df_iterator:
                image_name = row["image_path"]
                full_image_path = os.path.join(participant_folder_path, image_folder, image_name)
                # current_image = cv2.imread(full_image_path)
                yield full_image_path, difficulty_level

    else:
        # iterate over the labeled images folder and yield the image paths and their corresponding difficulty level
        labeled_images_path = participant_folder_path / labeled_images_folder
        for difficulty_level in os.listdir(labeled_images_path):
            # TODO only subset for faster testing
            for image_file in os.listdir(os.path.join(labeled_images_path, difficulty_level))[:subset_size]:
                full_image_path = os.path.join(labeled_images_path, difficulty_level, image_file)
                # current_image = cv2.imread(full_image_path)
                yield full_image_path, difficulty_level


class DifficultyLevels(Enum):
    HARD = "hard"
    MEDIUM = "medium"
    EASY = "easy"

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    @staticmethod
    def get_one_hot_encoding(category):
        if category not in DifficultyLevels.values():
            sys.stderr.write(f"\nNo one hot encoding possible for category {category}. Must be one of '"
                             f"{DifficultyLevels.values()}'!")
            return None

        # return one-hot-encoded vector for this category
        if category == DifficultyLevels.HARD.value:
            label_vector = [0, 0, 1]
        elif category == DifficultyLevels.MEDIUM.value:
            label_vector = [0, 1, 0]
        elif category == DifficultyLevels.EASY.value:
            label_vector = [1, 0, 0]
        else:
            sys.stderr.write(f"\nCategory {category} doesn't match one of the difficulty levels!")
            return None

        return label_vector

    @staticmethod
    def get_label_for_encoding(encoded_vector):
        if encoded_vector == [0, 0, 1]:
            label = DifficultyLevels.HARD.value
        elif encoded_vector == [0, 1, 0]:
            label = DifficultyLevels.MEDIUM.value
        elif encoded_vector == [1, 0, 0]:
            label = DifficultyLevels.EASY.value
        else:
            sys.stderr.write(f"\nEncoded_vector {encoded_vector} can't be matched to one of the labels!")
            return None

        return label


def preprocess_train_test_data(data):
    train_test_data = []

    for participant in data:
        for image_path, difficulty_level in get_participant_images(participant, use_folder=False):
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
                train_test_data.append([np.array(resized_img), label_vector, image_path])

            except Exception as e:
                sys.stderr.write(f"\nError in reading and resizing image '{image_path}': {e}")

    random.shuffle(train_test_data)
    return train_test_data


def start_preprocessing():
    all_participants = os.listdir(data_folder_path)  # [:12]  # TODO only take 12 so the counterbalancing works
    random.shuffle(all_participants)

    train_ratio = 0.8
    train_split = int(len(all_participants) * train_ratio)
    train_participants = all_participants[:train_split]
    test_participants = all_participants[train_split:]
    print("Number of participants used for training: ", len(train_participants))
    print("Number of participants used for testing: ", len(test_participants))

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

    show_batch([data[0] for data in train_data[25:]], [data[1] for data in train_data[25:]])

    train_images = np.asarray(train_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)
    test_images = np.asarray(test_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)
    # normalize all images to [0, 1] for the neural network
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    result_folder = "ml_results"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # TODO use savez to save compressed?
    np.save(os.path.join(result_folder, "train_images.npy"), train_images)
    np.save(os.path.join(result_folder, "train_labels.npy"), train_labels)
    np.save(os.path.join(result_folder, "train_paths.npy"), train_paths)

    np.save(os.path.join(result_folder, "test_images.npy"), test_images)
    np.save(os.path.join(result_folder, "test_labels.npy"), test_labels)
    np.save(os.path.join(result_folder, "test_paths.npy"), test_paths)


if __name__ == "__main__":
    start_preprocessing()
