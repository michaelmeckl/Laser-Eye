#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
from machine_learning_predictor.machine_learning_constants import NUMBER_OF_CLASSES, data_folder_path, results_folder, \
    NEW_IMAGE_SIZE, RANDOM_SEED

# for reproducibility set this BEFORE importing tensorflow: see
# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

import pathlib
import random
import pandas as pd
import tensorflow as tf
import numpy as np
import itertools
from sklearn import metrics
from matplotlib import pyplot as plt
from machine_learning_predictor.classifier import DifficultyImageClassifier
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.ml_utils import set_random_seed
from post_processing.post_processing_constants import download_folder
from post_processing.extract_downloaded_data import get_smallest_fps


def merge_participant_image_logs(participant_list, images_path, test_mode=True):
    image_data_frame = pd.DataFrame()

    for participant in participant_list:
        images_label_log = images_path / download_folder / participant / "labeled_images.csv"
        # images_label_log = images_path / download_folder / participant / "labeled_eye_regions.csv"
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


def get_suitable_sample_size(category_size):
    # use a divisor of the amount of images per difficulty category for a participant
    # -> this way their won't be any overlap of label categories or participants per sample!
    sample_size = 1
    for i in range(11, 101):
        if category_size % i == 0:
            sample_size = i
            break

    # TODO
    """
    # fps = get_smallest_fps()  # TODO
    fps = 14.3

    sample_time_span = 6  # 6 seconds as in the Fridman Paper: "Cognitive Load Estimation in the Wild"
    sample_size = round(fps * sample_time_span)  # the number of images we take as one sample
    """
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


def show_generator_example_images(batch, labels, sample_size, gen_v2=False):
    batch_len = len(batch)
    length = min(100, batch_len)  # show 100 images or batch length if one batch has less images

    img_height, img_width = NEW_IMAGE_SIZE

    if gen_v2:
        plt.figure(figsize=(10, 10))
        for i in range(length):
            # plt.subplot(1, batch_len, i + 1)  # if vertically stacked
            plt.subplot(batch_len, 1, i + 1)  # if horizontally stacked
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            # if vertically stacked:
            # plt.imshow(batch[i][0:sample_size * img_height, :, :])
            # plt.title(DifficultyLevels.get_label_for_encoding(labels[i]))

            # if horizontally stacked:
            plt.imshow(batch[i][:, 0:sample_size * img_width, :])
            plt.ylabel(DifficultyLevels.get_label_for_encoding(labels[i]))
    else:
        plt.figure(figsize=(10, 10))
        for i in range(length):
            plt.subplot(10, 10, i + 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(batch[i])
            plt.title(DifficultyLevels.get_label_for_encoding(labels[i]))

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    plt.savefig(os.path.join(results_folder, f"example_images_{'gen_2' if gen_v2 else 'gen_1'}.png"))
    plt.show()


def configure_for_performance(ds, filename=None):
    if filename:
        ds_folder = "cached_dataset"
        if not os.path.exists(ds_folder):
            os.mkdir(ds_folder)
        ds = ds.cache(os.path.join(ds_folder, filename))
        # Important: run through the dataset once to finalize the cache!
        print(list(ds.as_numpy_iterator()))
    else:
        # only cache in memory, not on disk
        ds = ds.cache()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_train_val_images():
    # without_participants = ["participant_1", "participant_5"]  # "participant_6"
    # for eye regions testing:
    # without_participants = ["participant_1", "participant_2", "participant_5", "participant_6",
    #                         "participant_7", "participant_8", "participant_9", "participant_12",
    #                         "participant_13"]
    without_participants = []

    all_participants = os.listdir(data_folder_path)[:12]  # only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    # split into train and test data
    train_participants, test_participants = split_train_test(all_participants)
    train_data = merge_participant_image_logs(train_participants, images_path)
    val_data = merge_participant_image_logs(test_participants, images_path)

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
    batch_size = 3

    sample_size = 30  # get_suitable_sample_size(difficulty_category_size)
    print(f"Sample size: {sample_size} (Train data len: {len(train_data)}, val data len: {len(val_data)})")

    use_gray = False  # TODO use grayscale images instead?
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
        train_dataset = configure_for_performance(train_dataset)
        val_dataset = configure_for_performance(val_dataset)
        """
        train_dataset = configure_for_performance(train_dataset, filename="train_cache")
        val_dataset = configure_for_performance(val_dataset, filename="val_cache")
        """
    return train_dataset, val_dataset


def train_classifier(train_generator, val_generator, batch_size, sample_size, train_epochs=15):
    image_shape = train_generator.get_image_shape()
    print("Image Shape: ", image_shape)

    classifier = DifficultyImageClassifier(train_generator, val_generator, num_classes=NUMBER_OF_CLASSES,
                                           num_epochs=train_epochs)
    batch, batch_labels = train_generator.get_example_batch()
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

    train_dataset, val_dataset = prepare_dataset(train_generator, val_generator, batch_size, sample_size, image_shape)
    classifier.try_transfer_ds_version(image_shape, train_dataset, val_dataset)

    # classifier.try_transfer(image_shape)
    return classifier


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Taken from https://www.tensorflow.org/tensorboard/image_summaries#building_an_image_classifier and slightly adjusted

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # row_sum = cm.sum(axis=1)[:, np.newaxis] if all(cm.sum(axis=1)) != 0 else [1, np.newaxis]
    # labels = np.around((cm.astype('float') / row_sum) if all(cm.sum(axis=1)) != 0 else cm.astype('float'), decimals=2)
    labels = np.around(cm.astype('float'), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(results_folder, "confusion_matrix.png"))
    plt.show()


def test_classifier(classifier, batch_size, sample_size):
    # get the participants that weren't used for training or validation
    test_participants = os.listdir(data_folder_path)[12:]
    print(f"Found {len(test_participants)} participants for testing.")

    random.shuffle(test_participants)
    test_df = merge_participant_image_logs(test_participants, images_path)

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

    # classifier.predict_test_generator(test_generator)

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
    label_names = DifficultyLevels.values()
    print(f"\nAccuracy score on test data: {metrics.accuracy_score(all_labels, all_predictions) * 100:.2f} %")
    print(f"Bal. accuracy score on test data: {metrics.balanced_accuracy_score(all_labels, all_predictions):.2f}")
    print(f"Precision score on test data: "
          f"{metrics.precision_score(all_labels, all_predictions, average='weighted')}:.2f")
    print(f"F1 score on test data: {metrics.f1_score(all_labels, all_predictions, average='weighted'):.2f}")
    print(f"\nClassification Report:\n"
          f"{metrics.classification_report(all_labels, all_predictions, target_names=label_names)}")

    # compute and show the confusion matrix
    conf_matrix = metrics.confusion_matrix(all_predictions, all_labels, normalize="all")
    print(f"Confusion Matrix:\n{conf_matrix}")
    plot_confusion_matrix(conf_matrix, label_names)


if __name__ == "__main__":
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

    use_gen_v2 = True
    use_dataset_version = False
    print(f"[INFO] Using custom generator version_2: {use_gen_v2}\n"
          f"[INFO] Using dataset version: {use_dataset_version}\n")

    if use_gen_v2:
        from machine_learning_predictor.custom_data_generator_v2 import CustomImageDataGenerator
    else:
        from machine_learning_predictor.custom_data_generator import CustomImageDataGenerator

    images_path = pathlib.Path(__file__).parent.parent / "post_processing"

    train_gen, val_gen, num_batches, num_samples = setup_data_generation(show_examples=False)
    difficulty_classifier = train_classifier(train_gen, val_gen, num_batches, num_samples)

    # test_classifier(difficulty_classifier, num_batches, num_samples)
