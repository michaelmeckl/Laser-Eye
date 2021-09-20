#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys

from machine_learning_predictor.machine_learning_constants import NUMBER_OF_CLASSES, data_folder_path, images_path, \
    RANDOM_SEED, TRAIN_EPOCHS, ml_data_folder

# for reproducibility set this BEFORE importing tensorflow: see
# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

import random
from enum import Enum
import pandas as pd
import tensorflow as tf
import numpy as np
from machine_learning_predictor.ml_utils import set_random_seed, split_train_test, show_generator_example_images, \
    calculate_prediction_results, get_suitable_sample_size
from post_processing.post_processing_constants import download_folder, post_processing_log_folder
from machine_learning_predictor.mixed_data_generator import MixedDataGenerator
from machine_learning_predictor.classifier import DifficultyImageClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector


############### Global vars
test_mode = True
test_subset_size = 250
###############


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def merge_participant_eye_tracking_logs(participant_list, dataset_type: DatasetType):
    blink_dataframe = pd.DataFrame()
    eye_log_dataframe = pd.DataFrame()

    for participant in participant_list:
        difficulty_dir_path = os.path.join(data_folder_path, participant, post_processing_log_folder)
        difficulty_dirs = os.listdir(difficulty_dir_path)

        for difficulty_dir in difficulty_dirs:
            for element in os.listdir(os.path.join(difficulty_dir_path, difficulty_dir)):
                if "blink_log" in element:
                    blink_log_path = os.path.join(difficulty_dir_path, difficulty_dir, element)
                    blink_log = pd.read_csv(blink_log_path)

                    # add two columns to the df
                    blink_log["participant"] = participant
                    blink_log["difficulty_level"] = difficulty_dir
                    blink_dataframe = pd.concat([blink_dataframe, blink_log])

                elif "processing_log" in element:
                    eye_log_path = os.path.join(difficulty_dir_path, difficulty_dir, element)
                    eye_log = pd.read_csv(eye_log_path, sep=";")
                    eye_log["participant"] = participant
                    eye_log["difficulty_level"] = difficulty_dir
                    if test_mode:
                        eye_log = eye_log[:test_subset_size]

                    eye_log_dataframe = pd.concat([eye_log_dataframe, eye_log])

            # participant_5 has 5 rows less for category "easy" than the rest after the eye tracking part so we simply
            # duplicate the last row in the dataframe so every participant has the same amount of data rows
            if not test_mode and difficulty_dir == "easy" and participant == "participant_5":
                eye_df_last_row = eye_log_dataframe.iloc[[-1]]
                for i in range(5):
                    eye_log_dataframe = eye_log_dataframe.append(eye_df_last_row)

    # add one-hot-encoding to the dataframes
    new_cols = pd.get_dummies(blink_dataframe['difficulty_level'], prefix='difficulty')
    blink_dataframe = pd.concat([blink_dataframe, new_cols], axis=1)

    new_cols = pd.get_dummies(eye_log_dataframe['difficulty_level'], prefix='difficulty')
    eye_log_dataframe = pd.concat([eye_log_dataframe, new_cols], axis=1)

    # reset the df index as the concatenate above creates duplicate indexes
    blink_dataframe_ordered = blink_dataframe.reset_index(drop=True)
    eye_log_dataframe_ordered = eye_log_dataframe.reset_index(drop=True)

    # shuffle the dataframe rows but maintain the row content,
    # see https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # blink_dataframe_ordered = blink_dataframe_ordered.sample(frac=1)

    # remove all '(', ')', '[' and ']' in the dataframe
    eye_log_dataframe_ordered.replace(to_replace=r"[\(\)\[\]]", value="", regex=True, inplace=True)

    # remove all trailing and leading whitespaces
    eye_log_dataframe_ordered.columns = eye_log_dataframe_ordered.columns.str.strip()
    cols = eye_log_dataframe_ordered.select_dtypes(['object']).columns
    eye_log_dataframe_ordered[cols] = eye_log_dataframe_ordered[cols].apply(lambda x: x.str.strip())

    # split columns so every column contains only one value
    eye_log_dataframe_ordered[['ROLL', 'PITCH', 'YAW']] = \
        eye_log_dataframe_ordered.HEAD_POS_ROLL_PITCH_YAW.str.split(", ", expand=True, )

    eye_log_dataframe_ordered[['LEFT_PUPIL_POS_X', 'LEFT_PUPIL_POS_Y']] = \
        eye_log_dataframe_ordered.LEFT_PUPIL_POS.str.split(" +", expand=True)  # " +" selects one or more whitespaces
    eye_log_dataframe_ordered[['RIGHT_PUPIL_POS_X', 'RIGHT_PUPIL_POS_Y']] = \
        eye_log_dataframe_ordered.RIGHT_PUPIL_POS.str.split(" +", expand=True)
    eye_log_dataframe_ordered[['LEFT_EYE_CENTER_X', 'LEFT_EYE_CENTER_Y']] = \
        eye_log_dataframe_ordered.LEFT_EYE_CENTER.str.split(" +", expand=True)
    eye_log_dataframe_ordered[['RIGHT_EYE_CENTER_X', 'RIGHT_EYE_CENTER_Y']] = \
        eye_log_dataframe_ordered.RIGHT_EYE_CENTER.str.split(" +", expand=True)

    # remove unnecessary columns
    eye_log_dataframe_ordered = eye_log_dataframe_ordered.drop(['date', 'HEAD_POS_ROLL_PITCH_YAW', 'LEFT_EYE_CENTER',
                                                                'RIGHT_EYE_CENTER', 'LEFT_PUPIL_POS', 'RIGHT_PUPIL_POS'
                                                                ], axis=1)
    # convert object data types back to floats
    string_dtypes = ['ROLL', 'PITCH', 'YAW', 'LEFT_PUPIL_POS_X', 'LEFT_PUPIL_POS_Y', 'RIGHT_PUPIL_POS_X',
                     'RIGHT_PUPIL_POS_Y', 'LEFT_EYE_CENTER_X', 'LEFT_EYE_CENTER_Y', 'RIGHT_EYE_CENTER_X',
                     'RIGHT_EYE_CENTER_Y']
    eye_log_dataframe_ordered[string_dtypes] = eye_log_dataframe_ordered[string_dtypes].astype(float)
    # print(eye_log_dataframe_ordered.dtypes)

    # save as csv files
    if dataset_type == DatasetType.TRAIN:
        eye_log_file = f"eye_log_dataframe_ordered_{DatasetType.TRAIN.value}.csv"
        blink_log_file = f"blink_dataframe_shuffled_{DatasetType.TRAIN.value}.csv"
    elif dataset_type == DatasetType.VALIDATION:
        eye_log_file = f"eye_log_dataframe_ordered_{DatasetType.VALIDATION.value}.csv"
        blink_log_file = f"blink_dataframe_shuffled_{DatasetType.VALIDATION.value}.csv"
    elif dataset_type == DatasetType.TEST:
        eye_log_file = f"eye_log_dataframe_ordered_{DatasetType.TEST.value}.csv"
        blink_log_file = f"blink_dataframe_shuffled_{DatasetType.TEST.value}.csv"
    else:
        print("Unknown Dataset Type given!")
        sys.exit(0)

    eye_log_dataframe_ordered.to_csv(os.path.join(ml_data_folder, eye_log_file), index=False)
    blink_dataframe_ordered.to_csv(os.path.join(ml_data_folder, blink_log_file), index=False)

    return blink_dataframe_ordered, eye_log_dataframe_ordered


def load_pupil_movement_data(participants, dataset_type: DatasetType):
    pupil_movement_dataframe = pd.DataFrame()
    pupil_movement_path = "pupil_movement_data_2"

    for participant in participants:
        # TODO this assumes that the csv files for each participant are in the correct difficulty order!!
        for csv_file in os.listdir(pupil_movement_path):
            csv_file_parts = csv_file.removesuffix(".csv").split("_")
            participant_name = f"{csv_file_parts[-3]}_{csv_file_parts[-2]}"  # extract the 'participant_xyz' part
            if participant == participant_name:
                csv_df = pd.read_csv(os.path.join(pupil_movement_path, csv_file))
                if test_mode:
                    csv_df = csv_df[:test_subset_size]

                pupil_movement_dataframe = pd.concat([pupil_movement_dataframe, csv_df])

    # add one-hot-encoding to the dataframe
    new_cols = pd.get_dummies(pupil_movement_dataframe['difficulty'], prefix='difficulty')
    pupil_movement_dataframe = pd.concat([pupil_movement_dataframe, new_cols], axis=1)

    pupil_movement_dataframe = pupil_movement_dataframe.reset_index(drop=True)

    if dataset_type == DatasetType.TRAIN:
        pupil_csv_file = f"{DatasetType.TRAIN.value}_pupil_movement.csv"
    elif dataset_type == DatasetType.VALIDATION:
        pupil_csv_file = f"{DatasetType.VALIDATION.value}_pupil_movement.csv"
    elif dataset_type == DatasetType.TEST:
        pupil_csv_file = f"{DatasetType.TEST.value}_pupil_movement.csv"
    else:
        print("Unknown Dataset Type given!")
        sys.exit(0)

    pupil_movement_dataframe.to_csv(os.path.join(ml_data_folder, pupil_csv_file), index=False)
    return pupil_movement_dataframe


def merge_participant_image_logs(participant_list, dataset_type: DatasetType):
    image_data_frame = pd.DataFrame()
    use_eye_regions = False

    # TODO get the difficulty order that was used for creating the eye and pupil log files to re-order the dataframe
    #  correctly so every row in the images df matches the correct row in the other dataframes
    # get path to post_processing sub folders of the first participant (as the order is the same for all participants)
    difficulty_dir_path = os.path.join(data_folder_path, "participant_1", post_processing_log_folder)
    row_order = os.listdir(difficulty_dir_path)  # the row order should be ["easy", "hard", "medium"]

    for participant in participant_list:
        if use_eye_regions:
            images_label_log = images_path / download_folder / participant / "labeled_eye_regions.csv"
        else:
            images_label_log = images_path / download_folder / participant / "labeled_images.csv"

        labeled_images_df = pd.read_csv(images_label_log)

        # participant_5 has 5 images less than the rest after the eye tracking part so we simply duplicate the last row
        if not test_mode and use_eye_regions and participant == "participant_5":
            last_row = labeled_images_df.iloc[[-1]]
            for i in range(5):
                labeled_images_df = labeled_images_df.append(last_row)

        difficulty_level_df = pd.DataFrame()
        # iterate over the difficulty levels in the correct order to create a correctly ordered dataframe
        for difficulty_level in row_order:
            # create a subset of the df that contains only the rows with this difficulty level
            sub_df = labeled_images_df[labeled_images_df.difficulty == difficulty_level]
            if test_mode:
                # for faster testing take only the first n rows for each difficulty level per participant
                sub_df = sub_df[:test_subset_size]

            difficulty_level_df = pd.concat([difficulty_level_df, sub_df])

        image_data_frame = pd.concat([image_data_frame, difficulty_level_df])

    # reset the df index as the concatenate above creates duplicate indexes
    image_data_frame_numbered = image_data_frame.reset_index(drop=True)
    # image_data_frame_numbered["index"] = image_data_frame_numbered.index  # add the index numbers as own column

    if dataset_type == DatasetType.TRAIN:
        image_csv_file = f"{DatasetType.TRAIN.value}_image_data_frame.csv"
    elif dataset_type == DatasetType.VALIDATION:
        image_csv_file = f"{DatasetType.VALIDATION.value}_image_data_frame.csv"
    elif dataset_type == DatasetType.TEST:
        image_csv_file = f"{DatasetType.TEST.value}_image_data_frame.csv"
    else:
        print("Unknown Dataset Type given!")
        sys.exit(0)

    image_data_frame_numbered.to_csv(os.path.join(ml_data_folder, image_csv_file), index=False)
    return image_data_frame_numbered


def get_train_val_data(train_participants, val_participants):
    # TODO if anything changes remove the existing logs
    #  also don't use the first part, i.e. the cached data, if running cross - validation !!
    if os.path.exists(ml_data_folder):
        # load existing data from csv files
        print("[INFO] Using cached data")
        print("Loading csv data ...\n")
        eye_log_train_data = pd.read_csv(os.path.join(ml_data_folder, "eye_log_dataframe_ordered_train.csv"))
        eye_log_val_data = pd.read_csv(os.path.join(ml_data_folder, "eye_log_dataframe_ordered_val.csv"))
        blink_train_data = pd.read_csv(os.path.join(ml_data_folder, "blink_dataframe_shuffled_train.csv"))
        blink_val_data = pd.read_csv(os.path.join(ml_data_folder, "blink_dataframe_shuffled_val.csv"))

        pupil_move_train = pd.read_csv(os.path.join(ml_data_folder, "train_pupil_movement.csv"))
        pupil_move_val = pd.read_csv(os.path.join(ml_data_folder, "val_pupil_movement.csv"))

        train_image_data = pd.read_csv(os.path.join(ml_data_folder, "train_image_data_frame.csv"))
        val_image_data = pd.read_csv(os.path.join(ml_data_folder, "val_image_data_frame.csv"))
    else:
        # generate the data new
        print("Generating csv data ...\n")
        os.mkdir(ml_data_folder)

        train_image_data = merge_participant_image_logs(train_participants, DatasetType.TRAIN)
        val_image_data = merge_participant_image_logs(val_participants, DatasetType.VALIDATION)

        blink_train_data, eye_log_train_data = merge_participant_eye_tracking_logs(train_participants,
                                                                                   DatasetType.TRAIN)
        blink_val_data, eye_log_val_data = merge_participant_eye_tracking_logs(val_participants,
                                                                               DatasetType.VALIDATION)
        pupil_move_train = load_pupil_movement_data(train_participants, DatasetType.TRAIN)
        pupil_move_val = load_pupil_movement_data(val_participants, DatasetType.VALIDATION)

    return train_image_data, val_image_data, eye_log_train_data, eye_log_val_data, pupil_move_train, pupil_move_val


def check_data(train_image_data, val_image_data, eye_log_train_data, eye_log_val_data):
    print("\n[Data Information]:")
    # make sure we have the same number of images per difficulty level!
    for difficulty_level in train_image_data.difficulty.unique():
        difficulty_level_df = train_image_data[train_image_data.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} train images for category \"{difficulty_level}\".")
    for difficulty_level in val_image_data.difficulty.unique():
        difficulty_level_df = val_image_data[val_image_data.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} val images for category \"{difficulty_level}\".")

    # make sure we have the same number of eye log rows per difficulty level!
    for difficulty_level in eye_log_train_data.difficulty_level.unique():
        difficulty_level_df = eye_log_train_data[eye_log_train_data.difficulty_level == difficulty_level]
        print(f"Found {len(difficulty_level_df)} train eye log rows for category \"{difficulty_level}\".")
    for difficulty_level in eye_log_val_data.difficulty_level.unique():
        difficulty_level_df = eye_log_val_data[eye_log_val_data.difficulty_level == difficulty_level]
        print(f"Found {len(difficulty_level_df)} val eye log rows for category \"{difficulty_level}\".")

    difficulty_category_size = None  # the amount of entries per difficulty category in the dataframe (the same for all)
    # make sure we have the same number of images per participant!
    for participant in train_image_data.participant.unique():
        participant_df = train_image_data[train_image_data.participant == participant]
        print(f"\nFound {len(participant_df)} train images for participant \"{participant}\".")

        if difficulty_category_size is None:
            # get the length of the first category for this participant (should be the same for all participants)
            for difficulty_level in participant_df.difficulty.unique():
                difficulty_level_df = participant_df[participant_df.difficulty == difficulty_level]
                difficulty_category_size = len(difficulty_level_df)
                break

    for participant in val_image_data.participant.unique():
        participant_df = val_image_data[val_image_data.participant == participant]
        print(f"Found {len(participant_df)} val images for participant \"{participant}\".\n")

    return difficulty_category_size


def setup_data_generation(train_participants, val_participants, show_examples=False):
    train_image_data, val_image_data, eye_log_train_data, eye_log_val_data, pupil_move_train, pupil_move_val = get_train_val_data(train_participants, val_participants)

    difficulty_category_size = check_data(train_image_data, val_image_data, eye_log_train_data, eye_log_val_data)

    # scale & standardize numerical data first,
    # see https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn
    # TODO select the correct columns for standard scaler!
    feature_columns = ['left_pupil_movement_x', 'left_pupil_movement_y', 'right_pupil_movement_x',
                       'right_pupil_movement_y', 'average_pupil_movement_x', 'average_pupil_movement_y',
                       'average_pupil_movement_distance', 'movement_angle']
    scaler = StandardScaler()
    # only fit once (on the training data) and use the fitted scaler on the validation and test data
    pupil_move_train[feature_columns] = scaler.fit_transform(pupil_move_train[feature_columns])
    pupil_move_val[feature_columns] = scaler.transform(pupil_move_val[feature_columns])

    # See https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network for consequences of
    # the batch size. Smaller batches lead to better results in general. Batch sizes are usually a power of two.
    batch_size = 4

    sample_size = get_suitable_sample_size(difficulty_category_size)  # TODO
    print(f"Sample size: {sample_size} (Train data len: {len(train_image_data)}, val data len: {len(val_image_data)})")

    use_gray = False
    train_generator = MixedDataGenerator(img_data_frame=train_image_data, eye_data_frame=pupil_move_train,
                                         x_col_name="image_path", y_col_name="difficulty",
                                         sequence_length=sample_size, batch_size=batch_size,
                                         images_base_path=images_path, use_grayscale=use_gray, is_train_set=True)

    val_generator = MixedDataGenerator(img_data_frame=val_image_data, eye_data_frame=pupil_move_val,
                                       x_col_name="image_path", y_col_name="difficulty",
                                       sequence_length=sample_size, batch_size=batch_size,
                                       images_base_path=images_path, use_grayscale=use_gray, is_train_set=False)

    if show_examples:
        # show some example train images to verify the generator is working correctly
        img_batch, eye_data_batch, batch_labels = train_generator.get_example_batch()
        show_generator_example_images(img_batch, batch_labels, sample_size, gen_v2=True)

    print("Len train generator: ", train_generator.__len__())
    print("Len val generator: ", val_generator.__len__())
    return train_generator, val_generator, batch_size, sample_size


def train_classifier(train_generator, val_generator, batch_size, sample_size, train_epochs=TRAIN_EPOCHS):
    image_shape = train_generator.get_image_shape()
    print("[INFO] Using image Shape: ", image_shape)
    eye_log_shape = train_generator.get_eye_log_shape()
    print("[INFO] Using eye_log Shape: ", eye_log_shape)

    classifier = DifficultyImageClassifier(train_generator, val_generator, num_classes=NUMBER_OF_CLASSES,
                                           num_epochs=train_epochs)

    train_history, val_accuracy = classifier.build_mixed_model(img_input_shape=image_shape,
                                                               eye_log_input_shape=eye_log_shape)
    # classifier.evaluate_classifier()

    return classifier, val_accuracy


def test_classifier(classifier, batch_size, sample_size):
    # get the participants that weren't used for training or validation
    test_participants = os.listdir(data_folder_path)[16:]
    print(f"Found {len(test_participants)} participants for testing.")

    random.shuffle(test_participants)

    # load test data
    if os.path.exists(ml_data_folder):
        print("[INFO] Using cached test data")
        image_test_df = pd.read_csv(os.path.join(ml_data_folder, "test_image_data_frame.csv"))
        eye_log_test_df = pd.read_csv(os.path.join(ml_data_folder, "eye_log_dataframe_ordered_test.csv"))
    else:
        print("Generating test data ...")
        image_test_df = merge_participant_image_logs(test_participants, DatasetType.TEST)
        eye_log_test_df = merge_participant_eye_tracking_logs(test_participants, DatasetType.TEST)

    for difficulty_level in image_test_df.difficulty.unique():
        difficulty_level_df = image_test_df[image_test_df.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} test images for category \"{difficulty_level}\".")

    for participant in image_test_df.participant.unique():
        participant_df = image_test_df[image_test_df.participant == participant]
        print(f"Found {len(participant_df)} test images for participant \"{participant}\".")

    use_gray = False
    test_generator = MixedDataGenerator(img_data_frame=image_test_df, eye_data_frame=eye_log_test_df,
                                        x_col_name="image_path", y_col_name="difficulty",
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


def start_training_and_testing_mixed_model():
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

    without_participants = []
    all_participants = os.listdir(data_folder_path)[:18]  # only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    train_participants, val_participants = split_train_test(all_participants)
    train_gen, val_gen, num_batches, num_samples = setup_data_generation(train_participants, val_participants)
    difficulty_classifier, val_accuracy = train_classifier(train_gen, val_gen, num_batches, num_samples)

    # use code below for cross-validation instead of the 3 lines above:
    """
    import gc

    all_accuracies = []
    n_splits = 5
    for i in range(n_splits):
        print(f"\n################## Starting split {i} / {n_splits} ################## \n")
        train_participants, val_participants = split_train_test(all_participants)  # shuffle and split into train & val
        # train_participants = (np.array(all_participants)[train_indices]).tolist()
        # val_participants = (np.array(all_participants)[test_indices]).tolist()

        train_gen, val_gen, num_batches, num_samples = setup_data_generation(train_participants, val_participants)
        difficulty_classifier, val_accuracy = train_classifier(train_gen, val_gen, num_batches, num_samples)

        all_accuracies.append(val_accuracy)
        print(f"Validation accuracy for split {i}: {val_accuracy * 100:.2f} %\n")
        gc.collect()  # manually call garbage collector at the end of each run to prevent OutOfMemory - Errors on GPU

    print(f"Mean accuracy over all splits: {np.mean(all_accuracies):.2f}")
    print(f"Best accuracy over all splits: {np.max(all_accuracies):.2f}")
    """

    # test_classifier(difficulty_classifier, num_batches, num_samples)


if __name__ == "__main__":
    start_training_and_testing_mixed_model()
