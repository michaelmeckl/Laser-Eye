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
from machine_learning_predictor.ml_utils import set_random_seed, split_train_test, show_generator_example_images, \
    configure_dataset_for_performance, calculate_prediction_results
from post_processing.post_processing_constants import download_folder


#################### Global settings #################
use_gen_v2 = True
use_dataset_version = False

if use_gen_v2:
    from machine_learning_predictor.custom_data_generator_v2 import CustomImageDataGenerator
else:
    from machine_learning_predictor.custom_data_generator import CustomImageDataGenerator

use_transfer_learning = True
if use_transfer_learning:
    from machine_learning_predictor.transfer_learning_classifier import DifficultyImageClassifier
else:
    from machine_learning_predictor.classifier import DifficultyImageClassifier

####################################


def merge_participant_image_logs(participant_list, test_mode=True):
    image_data_frame = pd.DataFrame()

    for participant in participant_list:
        images_label_log = images_path / download_folder / participant / "labeled_images.csv"
        # images_label_log = images_path / download_folder / participant / "labeled_eye_regions.csv"
        labeled_images_df = pd.read_csv(images_label_log)

        if test_mode:
            # for faster testing take only the first 150 rows for each difficulty level per participant
            test_subset_size = 250

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

    all_participants = os.listdir(data_folder_path)[:18]  # only take 12 or 18 so the counterbalancing works
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

    for difficulty_level in val_data.difficulty.unique():
        difficulty_level_df = val_data[val_data.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} val images for category \"{difficulty_level}\".")

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

    for participant in val_data.participant.unique():
        participant_df = val_data[val_data.participant == participant]
        print(f"Found {len(participant_df)} val images for participant \"{participant}\".")

    # See https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network for consequences of
    # the batch size. Smaller batches lead to better results in general. Batch sizes are usually a power of two.
    batch_size = 4

    sample_size = 25  # get_suitable_sample_size(difficulty_category_size)  # TODO
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
        batch, batch_labels = train_generator.get_example_batch()
        show_generator_example_images(batch, batch_labels, sample_size, gen_v2=use_gen_v2)
        """
        batch, batch_labels = train_generator.get_example_batch(2)
        show_generator_example_images(batch, batch_labels, sample_size, gen_v2=use_gen_v2)

        batch, batch_labels = train_generator.get_example_batch(3)
        show_generator_example_images(batch, batch_labels, sample_size, gen_v2=use_gen_v2)

        batch, batch_labels = train_generator.get_example_batch(4)
        show_generator_example_images(batch, batch_labels, sample_size, gen_v2=use_gen_v2)

        batch, batch_labels = train_generator.get_example_batch(idx=5)
        show_generator_example_images(batch, batch_labels, sample_size, gen_v2=use_gen_v2)
        """

    print("Len train generator: ", train_generator.__len__())
    print("Len val generator: ", val_generator.__len__())
    return train_generator, val_generator, batch_size, sample_size


def prepare_dataset(train_generator, val_generator, batch_size, sample_size, image_shape):
    if use_gen_v2:
        ds_output_signature = (
            tf.TensorSpec(shape=(batch_size, *image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, NUMBER_OF_CLASSES), dtype=tf.float32),
        )
    else:
        ds_output_signature = (
            tf.TensorSpec(shape=((sample_size * batch_size), *image_shape), dtype=tf.float32),
            tf.TensorSpec(shape=((sample_size * batch_size), NUMBER_OF_CLASSES), dtype=tf.float32),
        )

    # make sure all the dataset preprocessing is done on the CPU so the GPU can fully be used for training
    # see https://cs230.stanford.edu/blog/datapipeline/#best-practices
    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_generator(lambda: train_generator,
                                                       output_signature=ds_output_signature)
        val_dataset = tf.data.Dataset.from_generator(lambda: val_generator,
                                                     output_signature=ds_output_signature)

        # add caching and prefetching to speed up the process
        print("Configuring dataset for performance ...")
        train_dataset = configure_dataset_for_performance(train_dataset)
        val_dataset = configure_dataset_for_performance(val_dataset)
        """
        train_dataset = configure_for_performance(train_dataset, filename="train_cache")
        val_dataset = configure_for_performance(val_dataset, filename="val_cache")
        """
    return train_dataset, val_dataset


def train_classifier(train_generator, val_generator, batch_size, sample_size, train_epochs=TRAIN_EPOCHS):
    image_shape = train_generator.get_image_shape()
    print("[INFO] Using image Shape: ", image_shape)

    classifier = DifficultyImageClassifier(train_generator, val_generator, num_classes=NUMBER_OF_CLASSES,
                                           num_epochs=train_epochs)
    # batch, batch_labels = train_generator.get_example_batch()
    # classifier.build_model(input_shape=image_shape, img_batch=batch)
    #
    # if use_dataset_version:
    #     train_dataset, val_dataset = prepare_dataset(train_generator, val_generator, batch_size, sample_size,
    #                                                  image_shape)
    #     classifier.train_classifier_dataset_version(train_dataset, val_dataset)
    #     classifier.evaluate_classifier_dataset_version(val_dataset)
    # else:
    #     classifier.train_classifier()
    #     classifier.evaluate_classifier()

    # train_dataset, val_dataset = prepare_dataset(train_generator, val_generator, batch_size, sample_size, image_shape)
    # classifier.try_transfer_ds_version(image_shape, train_dataset, val_dataset)

    classifier.train_transfer_model(image_shape)
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


def start_training_and_testing():
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

    print(f"[INFO] Using custom generator version_2: {use_gen_v2}\n"
          f"[INFO] Using dataset version: {use_dataset_version}\n")

    print(f"[INFO] Using transfer learning: {use_transfer_learning}\n")

    train_gen, val_gen, num_batches, num_samples = setup_data_generation(show_examples=False)
    difficulty_classifier = train_classifier(train_gen, val_gen, num_batches, num_samples)

    # test_classifier(difficulty_classifier, num_batches, num_samples)


if __name__ == "__main__":
    start_training_and_testing()
