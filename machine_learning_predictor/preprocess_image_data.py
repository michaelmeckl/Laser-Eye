#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


NEW_IMAGE_SIZE = 100
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


def preprocess_train_test_data(data, is_test_data=False):
    train_test_data = []

    for participant in data:
        image_folder = os.path.join(data_folder_path, participant, "labeled_images")
        path_high_load = os.path.join(image_folder, "hard")
        path_normal_load = os.path.join(image_folder, "medium")
        path_low_load = os.path.join(image_folder, "easy")

        categories = [path_low_load, path_normal_load, path_high_load]
        for category_path in categories:
            print(f"Processing category: {category_path}")

            # one-hot-encoding
            if category_path is path_high_load:
                label = [0, 0, 1]
            elif category_path is path_normal_load:
                label = [0, 1, 0]
            else:
                label = [1, 0, 0]

            # TODO für testen nur Teil der Daten nehmen
            all_images_for_category = os.listdir(f"{category_path}")[:250]

            for img in all_images_for_category:
                path_to_image = os.path.join(category_path, img)

                try:
                    grayscale_img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
                    cv2.imshow("grayscale", grayscale_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    resized_img = cv2.resize(grayscale_img, (NEW_IMAGE_SIZE, NEW_IMAGE_SIZE))

                    # if is_test_data:
                    train_test_data.append([np.array(resized_img), label, path_to_image])

                except Exception as e:
                    print(f"Error in resizing image '{img}': {e}")

    random.shuffle(train_test_data)
    return train_test_data


def start_preprocessing():
    all_participants = os.listdir(data_folder_path)
    random.shuffle(all_participants)

    train_ratio = 0.8
    train_split = int(len(all_participants) * train_ratio)
    train_participants = all_participants[:train_split]
    test_participants = all_participants[train_split:]

    train_data = preprocess_train_test_data(train_participants, is_test_data=False)
    test_data = preprocess_train_test_data(test_participants, is_test_data=True)

    print(len(train_data))
    print(len(test_data))
    # show_batch([data[0] for data in train_data[25:]])

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

    train_images = np.asarray(train_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)
    test_images = np.asarray(test_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    result_folder = "ml_results"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    np.save(os.path.join(result_folder, "train_images.npy"), train_images)
    np.save(os.path.join(result_folder, "train_labels.npy"), train_labels)
    np.save(os.path.join(result_folder, "train_paths.npy"), train_paths)

    np.save(os.path.join(result_folder, "test_images.npy"), test_images)
    np.save(os.path.join(result_folder, "test_labels.npy"), test_labels)
    np.save(os.path.join(result_folder, "test_paths.npy"), test_paths)


if __name__ == "__main__":
    start_preprocessing()
