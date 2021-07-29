#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import random
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
def load_batched_data():
    # TODO needs to be in subdirectories definingt the class labels
    train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_images = train_generator.flow_from_directory(data_folder_path, target_size=(150, 150), batch_size=32,
                                                       class_mode='categorical')

    image_batch, label_batch = next(train_images)
    show_batch(image_batch, label_batch)


def split_train_test(data):
    # train + validation & test data
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.33, random_state = 1,
                                                                    stratify = y)

    # make a copy
    X_train_valid = X_train_valid.copy()
    X_test = X_test.copy()

    # train & validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,
                                                          y_train_valid,
                                                          test_size=0.3,
                                                          random_state=42)
    X_train = X_train.copy()
    X_valid = X_valid.copy()


def one_hot_encode_labels(y_train, y_test):
    from numpy import argmax
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    # define example
    # data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
    values = np.array(y_test)
    print(values)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)  # keras does not like sparse arrays
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(inverted)


def scale_data():
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # ! only use fit_transform once at the start as this trains the model!
    X_test = scaler.transform(X_test)
"""


NEW_IMAGE_SIZE = 150
data_folder_path = os.path.join(os.path.dirname(__file__), "..", "post_processing", "tracking_data_download")

random.seed(42)  # for reproducibility


def show_batch(image_batch, label_batch=None):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(f"{label_batch}")
        plt.axis('off')
    plt.show()


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    Taken from https://stackoverflow.com/a/312464
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_fps_info(fps_file_path):
    with open(fps_file_path, mode="r") as fps_file:
        lines = fps_file.readlines()
        fps_line = lines[1]  # the fps is in the second line
        fps = re.findall(r"\d+\.?\d+", fps_line)
        fps_rounded = round(float(fps[0]), ndigits=2)

    return fps_rounded


def preprocess_train_test_data(data, is_test_data=False):
    data_batches = []
    train_test_data = []

    for participant in data:
        fps_log_path = os.path.join(data_folder_path, participant, "fps_info.txt")
        fps = get_fps_info(fps_log_path)
        # print(fps)

        batch_time_span = 6  # 6 seconds as in the Fridman Paper: "Cognitive Load Estimation in the Wild"
        batch_size = round(fps * batch_time_span)  # the number of images we take as one batch

        image_folder = os.path.join(data_folder_path, participant, "labeled_images")
        path_high_load = os.path.join(image_folder, "hard")
        path_normal_load = os.path.join(image_folder, "medium")
        path_low_load = os.path.join(image_folder, "easy")

        categories = [path_low_load, path_normal_load, path_high_load]
        random.shuffle(categories)
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

            # create batches
            """
            category_batches = []
            current_batch = []
            c = 0
            counter_overall = 0
            for image in all_images_for_category:
                if c == batch_size or counter_overall == len(all_images_for_category)-1:
                    print("Current batch size:", len(current_batch))
                    category_batches.append(current_batch)
                    current_batch = []
                    c = 0

                current_batch.append(image)
                c += 1
                counter_overall += 1

            random.shuffle(category_batches)
            """
            category_batches = [(chunk, label, category_path) for chunk in chunks(all_images_for_category, batch_size)]
            random.shuffle(category_batches)
            data_batches.extend(category_batches)

    # shuffle again so the participants are mixed up as well!
    random.shuffle(data_batches)

    for batch in data_batches:
        img_list, label, category_path = batch
        for img in img_list:
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

    return train_test_data


def start_preprocessing():
    """Idee:
    Featuremenge verkleinern (feature engineering??)
    1. Zeitspanne für batches basierend auf Literatur festlegen, z.B. 3 Sekunden ist ein Batch
    2. anhand der fps info pro Teilnehmer die Menge an Bilder für diesen Zeitraum berechnen
        - z.B. 30 fps und 12 fps
        - 30: 3 * 30 = 90 -> für diesen Teilnehmer ergeben alle 90 aufeinanderfolgenden Bilder einen 3s - Batch
        - 12: 3 * 12 = 36 -> für diesen Teilnehmer ergeben alle 36 aufeinanderfolgenden Bilder einen 3s - Batch
    3. diese batches zum split in train, validation und test set verwenden
    4. diese batches auch shufflen?

    5. Wie an NN übergeben? ganz normal nur die Bilder, weil die Zeit pro Batch ja implizit mit drin ist (?)

    Ablauf:
    1. Teilnehmer shufflen
    2. 80% train test split Teilnehmer
    3. pro Teilnehmer:
       1. einteilen in high, normal, low
       2. one-hot-encoding
       3. die 3 Kategorien shufflen
       4. pro Kategorie:
          1. batches berechnen und erzeugen
          2. Batches shufflen
    4. alle batches über alle Teilnehmer nochmal shufflen, damit die Teilnehmer Reihenfolge durcheinander ist (sonst reines Overfitting, ca. 20% validation accuracy)
    5. bilder / label / path an train data anhängen
    6. CNN
    7. ca. 99% auf train, 76% auf validation und 42% auf test
    """

    # TODO use only images of the eye region instead of the whole face?
    all_participants = os.listdir(data_folder_path)
    random.shuffle(all_participants)

    train_ratio = 0.8
    train_split = int(len(all_participants) * train_ratio)
    train_participants = all_participants[:train_split]
    test_participants = all_participants[train_split:]

    # TODO k-fold-cross-validation
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
    # print(train_images[:1])
    test_images = np.asarray(test_images).reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)

    train_images = train_images/255.0
    test_images = test_images/255.0
    # print("\nafter normalizing:\n " + str(train_images))

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
