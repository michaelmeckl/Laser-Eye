#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
from machine_learning_predictor.machine_learning_constants import NUMBER_OF_CLASSES, data_folder_path, images_path, \
    RANDOM_SEED, TRAIN_EPOCHS

# for reproducibility set this BEFORE importing tensorflow: see
# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

import random
import pandas as pd
import tensorflow as tf
import numpy as np
from machine_learning_predictor.ml_utils import set_random_seed, split_train_test, \
    calculate_prediction_results, get_suitable_sample_size
from post_processing.post_processing_constants import download_folder
from matplotlib import pyplot as plt

from machine_learning_predictor.time_series_generator import CustomImageDataGenerator
from machine_learning_predictor.time_series_classifier import DifficultyImageClassifier


def merge_participant_image_logs(participant_list, test_mode=False):
    image_data_frame = pd.DataFrame()

    for participant in participant_list:
        # images_label_log = images_path / download_folder / participant / "labeled_images.csv"
        images_label_log = images_path / download_folder / participant / "labeled_eye_regions.csv"
        labeled_images_df = pd.read_csv(images_label_log)

        if test_mode:
            # for faster testing take only the first 150 rows for each difficulty level per participant
            test_subset_size = 150

            difficulty_level_df = pd.DataFrame()
            for difficulty_level in labeled_images_df.difficulty.unique():
                # create a subset of the df that contains only the rows with this difficulty level
                sub_df = labeled_images_df[labeled_images_df.difficulty == difficulty_level]
                sub_df = sub_df[:test_subset_size]
                difficulty_level_df = pd.concat([difficulty_level_df, sub_df])

            image_data_frame = pd.concat([image_data_frame, difficulty_level_df])
        else:
            image_data_frame = pd.concat([image_data_frame, labeled_images_df])

    # reset the df index as the concatenate above creates duplicate indexes
    image_data_frame_numbered = image_data_frame.reset_index(drop=True)
    # image_data_frame_numbered["index"] = image_data_frame_numbered.index  # add the index numbers as own column

    return image_data_frame_numbered


def get_train_val_images():
    # without_participants = ["participant_1", "participant_5"]
    without_participants = []

    all_participants = os.listdir(data_folder_path)[:16]  # only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    # split into train and test data
    train_participants, test_participants = split_train_test(all_participants)
    train_data = merge_participant_image_logs(train_participants)
    val_data = merge_participant_image_logs(test_participants)

    return train_data, val_data


def setup_data_generation(show_examples=True):
    train_data, val_data = get_train_val_images()

    # make sure we have the same number of images per difficulty level!
    for difficulty_level in train_data.difficulty.unique():
        difficulty_level_df = train_data[train_data.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} train images for category \"{difficulty_level}\".")

    difficulty_category_size = None  # the amount of entries per difficulty category in the dataframe (the same for all)
    # make sure we have the same number of images per participant!
    for participant in train_data.participant.unique():
        participant_df = train_data[train_data.participant == participant]
        print(f"Found {len(participant_df)} train images for participant \"{participant}\".")

        if difficulty_category_size is None:
            # get the length of the first category for this participant (should be the same for all participants)
            for difficulty_level in participant_df.difficulty.unique():
                difficulty_level_df = participant_df[participant_df.difficulty == difficulty_level]
                difficulty_category_size = len(difficulty_level_df)
                break

    # See https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network for consequences of
    # the batch size. Smaller batches lead to better results in general. Batch sizes are usually a power of two.
    batch_size = 4

    sample_size = get_suitable_sample_size(difficulty_category_size)
    print(f"Sample size: {sample_size} (Train data len: {len(train_data)}, val data len: {len(val_data)})")

    use_gray = False
    train_generator = CustomImageDataGenerator(data_frame=train_data, x_col_name="image_path", y_col_name="difficulty",
                                               sequence_length=sample_size, batch_size=batch_size,
                                               images_base_path=images_path, use_grayscale=use_gray, is_train_set=True)

    val_generator = CustomImageDataGenerator(data_frame=val_data, x_col_name="image_path", y_col_name="difficulty",
                                             sequence_length=sample_size, batch_size=batch_size,
                                             images_base_path=images_path, use_grayscale=use_gray, is_train_set=False)

    if show_examples:
        # show some example train images to verify the generator is working correctly
        batch, labels = train_generator.get_example_batch()

        for j, image in enumerate(range(len(batch))):
            batch_len = len(batch[j])
            length = min(100, batch_len)

            plt.figure(figsize=(15, 10))
            for i in range(length):
                plt.subplot(10, 10, i + 1)
                plt.grid(False)
                plt.imshow(batch[j][i])
                plt.xlabel(labels[j])
                plt.xticks([])
                plt.yticks([])
            plt.show()

    print("Len train generator: ", train_generator.__len__())
    print("Len val generator: ", val_generator.__len__())
    return train_generator, val_generator, batch_size, sample_size


def train_classifier(train_generator, val_generator, batch_size, sample_size, train_epochs=TRAIN_EPOCHS):
    image_shape = train_generator.get_image_shape()
    print("[INFO] Using image Shape: ", image_shape)

    classifier = DifficultyImageClassifier(train_generator, val_generator, num_classes=NUMBER_OF_CLASSES,
                                           num_epochs=train_epochs)
    classifier.build_model(input_shape=image_shape)
    classifier.train_classifier()
    classifier.evaluate_classifier()
    return classifier


def test_classifier(classifier, batch_size, sample_size):
    # get the participants that weren't used for training or validation
    test_participants = os.listdir(data_folder_path)[16:]
    print(f"Found {len(test_participants)} participants for testing.")

    random.shuffle(test_participants)
    test_df = merge_participant_image_logs(test_participants)

    for difficulty_level in test_df.difficulty.unique():
        difficulty_level_df = test_df[test_df.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} test images for category \"{difficulty_level}\".")

    for participant in test_df.participant.unique():
        participant_df = test_df[test_df.participant == participant]
        print(f"Found {len(participant_df)} test images for participant \"{participant}\".")

    # batch_size = 3  # use a different batch size for prediction? Useful as sample size must be the same as in training
    use_gray = False
    test_generator = CustomImageDataGenerator(data_frame=test_df, x_col_name="image_path", y_col_name="difficulty",
                                              sequence_length=sample_size, batch_size=batch_size,
                                              images_base_path=images_path, use_grayscale=use_gray, is_train_set=False)

    all_predictions = np.array([])
    all_labels = np.array([])
    # take 3 random generator outputs for prediction
    # for choice in random.sample(range(test_generator.__len__()), k=3):
    for i in range(test_generator.__len__()):
        test_image_batch, labels = test_generator.get_example_batch(idx=i)
        # show_generator_example_images(test_image_batch, labels, sample_size, gen_v2=use_gen_v2)
        predictions = classifier.predict(test_image_batch, labels)

        predictions_results = np.argmax(predictions, axis=1)
        all_predictions = np.concatenate([all_predictions, predictions_results])
        actual_labels = np.argmax(labels, axis=1)
        all_labels = np.concatenate([all_labels, actual_labels])

    # show some result metrics
    calculate_prediction_results(all_labels, all_predictions)


def start_training_and_testing_convlstm():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # uncomment this to always use the CPU instead of a GPU

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("Found GPU:", gpu.name, "  Type:", gpu.device_type)

    if len(gpus) == 0:
        print("No gpu found!")
    else:
        # set this to make CUDA deterministic as well; see
        # https://stackoverflow.com/questions/60910157/do-i-need-to-set-seed-in-all-modules-where-i-import-numpy-or-tensorflow
        # However, using a GPU may still result in slightly different results for each run because of parallelism.
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        try:
            # prevent tensorflow from allocating all available memory on the physical device
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except (ValueError, RuntimeError):
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    set_random_seed()  # set seed for reproducibility

    train_gen, val_gen, num_batches, num_samples = setup_data_generation(show_examples=False)
    difficulty_classifier = train_classifier(train_gen, val_gen, num_batches, num_samples)

    test_classifier(difficulty_classifier, num_batches, num_samples)


if __name__ == "__main__":
    start_training_and_testing_convlstm()
