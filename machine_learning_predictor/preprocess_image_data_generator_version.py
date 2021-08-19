#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import pathlib
import random
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from machine_learning_predictor.classifier import DifficultyImageClassifier
from machine_learning_predictor.custom_data_generator import CustomImageDataGenerator
from machine_learning_predictor.machine_learning_constants import NUMBER_OF_CLASSES, data_folder_path
from machine_learning_predictor.ml_utils import set_random_seed, show_result_plot
from post_processing.post_processing_constants import download_folder


def merge_participant_image_logs(participant_list):
    image_data_frame = pd.DataFrame()
    post_processing_folder_path = pathlib.Path(__file__).parent.parent / "post_processing"

    for participant in participant_list:
        images_label_log = post_processing_folder_path / download_folder / participant / "labeled_images.csv"
        labeled_images_df = pd.read_csv(images_label_log)

        difficulty_level_df = pd.DataFrame()
        # TODO for testing take only the first 150 rows for each difficulty level
        for difficulty_level in labeled_images_df.difficulty.unique():
            # create a subset of the df that contains only the rows with this difficulty level
            sub_df = labeled_images_df[labeled_images_df.difficulty == difficulty_level]
            sub_df = sub_df[:150]
            difficulty_level_df = pd.concat([difficulty_level_df, sub_df])

        image_data_frame = pd.concat([image_data_frame, difficulty_level_df])

    # add the index numbers as own column (reset the index first as the concatenate above creates duplicate indexes)
    image_data_frame_numbered = image_data_frame.reset_index(drop=True)
    image_data_frame_numbered["index"] = image_data_frame_numbered.index

    # shuffle the dataframe rows but maintain the row content,
    # see https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # image_data_frame_numbered = image_data_frame_numbered.sample(frac=1)

    return image_data_frame_numbered


def get_suitable_sample_size():
    """
    fps_log_path = os.path.join(data_folder_path, participant_folder, "fps_info.txt")
    fps = get_fps_info(fps_log_path)
    print("FPS:", fps)
    """
    fps = 14  # TODO use the fps that is used for post processing as well and drop images for users with more!
    sample_time_span = 6  # 6 seconds as in the Fridman Paper: "Cognitive Load Estimation in the Wild"
    sample_size = round(fps * sample_time_span)  # the number of images we take as one sample
    print("Sample size: ", sample_size)
    return sample_size


def split_train_test(participant_list, train_ratio=0.8):
    random.shuffle(participant_list)

    train_split = int(len(participant_list) * train_ratio)
    train_participants = participant_list[:train_split]
    test_participants = participant_list[train_split:]
    print(f"{len(train_participants)} participants used for training: {train_participants}")
    print(f"{len(test_participants)} participants used for validation: {test_participants}")

    return train_participants, test_participants


def configure_for_performance(ds, filename, batch_size):
    ds_folder = "cached_dataset"
    if not os.path.exists(ds_folder):
        os.mkdir(ds_folder)
    ds = ds.cache(os.path.join(ds_folder, filename))
    # TODO batch based on train or val batch size to increase performance further? -> this adds another dimension!
    # ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def start_preprocessing(use_dataset_version=True):
    set_random_seed()  # set seed for reproducibility

    # without_participants = ["participant_1", "participant_5"]  # "participant_6"
    without_participants = []

    all_participants = os.listdir(data_folder_path)[:12]  # TODO only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    train_participants, test_participants = split_train_test(all_participants)

    train_data = merge_participant_image_logs(train_participants)
    val_data = merge_participant_image_logs(test_participants)

    # See https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network for consequences of
    # the batch size. Smaller batches lead to better results in general. Batch sizes are usually a power of two.
    batch_size = 64

    sample_size = get_suitable_sample_size()
    print(f"Sample size: {sample_size} (Train data len: {len(train_data)}, val data len: {len(val_data)})")

    # TODO make sure we have nearly the same number of images per difficulty level!
    for difficulty_level in train_data.difficulty.unique():
        difficulty_level_df = train_data[train_data.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} train images for category \"{difficulty_level}\".")

    images_path = pathlib.Path(__file__).parent.parent / "post_processing"
    use_gray = False
    train_generator = CustomImageDataGenerator(data_frame=train_data, x_col_name="image_path", y_col_name="difficulty",
                                               sample_size=sample_size, batch_size=batch_size,
                                               images_base_path=images_path, use_grayscale=use_gray)

    val_generator = CustomImageDataGenerator(data_frame=val_data, x_col_name="image_path", y_col_name="difficulty",
                                             sample_size=sample_size, batch_size=batch_size,
                                             images_base_path=images_path, use_grayscale=use_gray)

    # show some example train images to verify the generator is working correctly
    first_sample, sample_labels = train_generator[0]
    sample_len = len(first_sample)
    length = min(25, sample_len)  # show 25 images or sample length if samples has less images

    plt.figure(figsize=(10, 10))
    for i in range(length):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(first_sample[i], cmap=plt.cm.binary)
        plt.xlabel(sample_labels[i])
    # plt.show()

    train_epochs = 12
    image_shape = train_generator.get_image_shape()
    print("Image Shape: ", image_shape)

    classifier = DifficultyImageClassifier(train_generator, val_generator, num_classes=NUMBER_OF_CLASSES,
                                           num_epochs=train_epochs)

    if use_dataset_version:
        ds_output_signature = (
            # tf.TensorSpec(shape=(batch_size, sample_size, *image_shape), dtype=tf.float64),
            tf.TensorSpec(shape=(sample_size, *image_shape), dtype=tf.float64),
            tf.TensorSpec(shape=(sample_size, NUMBER_OF_CLASSES), dtype=tf.float64),
        )
        train_dataset = tf.data.Dataset.from_generator(lambda: train_generator, output_signature=ds_output_signature)
        val_dataset = tf.data.Dataset.from_generator(lambda: val_generator, output_signature=ds_output_signature)

        # add caching and prefetching to speed up the process
        train_dataset = configure_for_performance(train_dataset, filename="train_dataset", batch_size=batch_size)
        val_dataset = configure_for_performance(val_dataset, filename="val_dataset", batch_size=batch_size)

        # TODO after adding batches:
        # new_image_shape = (batch_size, sample_size, *image_shape)
        # print("New image shape: ", new_image_shape)

        classifier.build_model(input_shape=image_shape)
        classifier.train_classifier_dataset_version(train_dataset, val_dataset)
        classifier.evaluate_classifier_dataset_version(val_dataset)

    else:
        classifier.build_model(input_shape=image_shape)
        classifier.train_classifier()
        classifier.evaluate_classifier()


if __name__ == "__main__":
    start_preprocessing()
