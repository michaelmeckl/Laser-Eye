#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import NUMBER_OF_CLASSES, data_folder_path, \
    RANDOM_SEED, TRAIN_EPOCHS

# for reproducibility set this BEFORE importing tensorflow: see
# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

import random
import pandas as pd
import tensorflow as tf
import numpy as np
from machine_learning_predictor.ml_utils import set_random_seed, split_train_test
from post_processing.post_processing_constants import post_processing_log_folder

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=400)


def merge_participant_eye_tracking_logs(participant_list):
    blink_dataframe = pd.DataFrame()
    eye_log_dataframe = pd.DataFrame()

    for participant in participant_list:
        difficulty_dir_path = os.path.join(data_folder_path, participant, post_processing_log_folder)
        difficulty_dirs = os.listdir(difficulty_dir_path)
        random.shuffle(difficulty_dirs)  # shuffle difficulty dirs so it won't be the same order for each participant

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
                    eye_log_dataframe = pd.concat([eye_log_dataframe, eye_log])

    # reset the df index as the concatenate above creates duplicate indexes
    blink_dataframe_ordered = blink_dataframe.reset_index(drop=True)
    eye_log_dataframe_ordered = eye_log_dataframe.reset_index(drop=True)

    # shuffle the dataframe rows but maintain the row content,
    # see https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    blink_dataframe_shuffled = blink_dataframe_ordered.sample(frac=1)

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

    return blink_dataframe_shuffled, eye_log_dataframe_ordered


def get_train_val_data():
    without_participants = []

    all_participants = os.listdir(data_folder_path)[:18]  # only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    # split into train and test data
    train_participants, test_participants = split_train_test(all_participants)
    blink_train_data, eye_log_train_data = merge_participant_eye_tracking_logs(train_participants)
    blink_val_data, eye_log_val_data = merge_participant_eye_tracking_logs(test_participants)

    return eye_log_train_data, blink_train_data, eye_log_val_data, blink_val_data


def setup_data_generation():
    eye_log_train_data, blink_train_data, eye_log_val_data, blink_val_data = get_train_val_data()

    # make sure we have the same number of images per difficulty level!
    for difficulty_level in eye_log_train_data.difficulty_level.unique():
        difficulty_level_df = eye_log_train_data[eye_log_train_data.difficulty_level == difficulty_level]
        print(f"Found {len(difficulty_level_df)} train samples for category \"{difficulty_level}\".")

    for difficulty_level in eye_log_val_data.difficulty_level.unique():
        difficulty_level_df = eye_log_val_data[eye_log_val_data.difficulty_level == difficulty_level]
        print(f"Found {len(difficulty_level_df)} val samples for category \"{difficulty_level}\".")

    difficulty_category_size = None  # the amount of entries per difficulty category in the dataframe (the same for all)
    # make sure we have the same number of images per participant!
    for participant in eye_log_train_data.participant.unique():
        participant_df = eye_log_train_data[eye_log_train_data.participant == participant]
        print(f"Found {len(participant_df)} train samples for participant \"{participant}\".")

        if difficulty_category_size is None:
            # get the length of the first category for this participant (should be the same for all participants)
            for difficulty_level in participant_df.difficulty_level.unique():
                difficulty_level_df = participant_df[participant_df.difficulty_level == difficulty_level]
                difficulty_category_size = len(difficulty_level_df)
                break

    for participant in eye_log_val_data.participant.unique():
        participant_df = eye_log_val_data[eye_log_val_data.participant == participant]
        print(f"Found {len(participant_df)} val samples for participant \"{participant}\".")

    # See https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network for consequences of
    # the batch size. Smaller batches lead to better results in general. Batch sizes are usually a power of two.
    batch_size = 4

    sample_size = 25  # get_suitable_sample_size(difficulty_category_size)
    print(f"Sample size: {sample_size}")

    print(eye_log_train_data.corr())
    print(eye_log_train_data.describe())

    merkmalsvektoren = {
        "klein": ["LEFT_PUPIL_DIAMETER", "RIGHT_PUPIL_DIAMETER"],
        "mittel": [""],
        "alle": [""],
    }
    X_train = eye_log_train_data[merkmalsvektoren["klein"]].values
    X_val = eye_log_val_data[merkmalsvektoren["klein"]].values
    y_train = eye_log_train_data["difficulty_level"].values
    y_val = eye_log_val_data["difficulty_level"].values

    plt.scatter(*X_val[y_val == "hard"].T, label="hard", c='steelblue')
    plt.scatter(*X_val[y_val == "medium"].T, label="medium", c='red')
    plt.scatter(*X_val[y_val == "easy"].T, label="easy", c='orange')
    plt.xlabel("LEFT_PUPIL_DIAMETER")
    plt.ylabel("RIGHT_PUPIL_DIAMETER")
    plt.legend()
    plt.show()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # TODO encode as 0, 1, 2 ?
    """
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_valid = encoder.transform(y_val)
    """

    # TODO
    # transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [0, 1])])
    # pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    # pipeline.fit(train_X, train_y)

    """
    from sklearn.linear_model import Perceptron
    perceptron = Perceptron(random_state=RANDOM_SEED)
    # trainieren
    perceptron.fit(X_train, y_train)
    # testen
    prediction_perceptron = perceptron.predict(X_val)
    # print(prediction_perceptron)
    print(f"Accuracy: {metrics.accuracy_score(y_val, prediction_perceptron) :.2%} correctly classified!")
    print(
        f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_val, prediction_perceptron):.2%} correctly classified!")
    """

    from sklearn.metrics import confusion_matrix
    class_names = DifficultyLevels.values()
    # cm = confusion_matrix(y_val, prediction_perceptron, normalize="all")
    # print(cm)
    # plot_confusion_matrix(cm, class_names)

    from sklearn.svm import SVC

    svc = SVC(kernel="linear", C=10, random_state=RANDOM_SEED)

    svc.fit(X_train, y_train)
    prediction_svm = svc.predict(X_val)
    print(f"Accuracy: {metrics.accuracy_score(y_val, prediction_svm) :.2%} correctly classified!")
    print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_val, prediction_svm):.2%} correctly classified!")

    # svm_plot(svc, X_val, y_val)


def svm_plot(svm_classifier, X, y):
    colors = ["red", "blue", "orange"]
    for i, reference_class in enumerate(np.unique(y)):
        plt.scatter(X[y == reference_class, 0], X[y == reference_class, 1], c=colors[i], label=str(
            reference_class), zorder=2)

    n = svm_classifier.coef_
    d = svm_classifier.intercept_
    x_0 = np.array((-1, 1))

    x_1 = (-x_0 * n[0, 0] - d) / n[0, 1]
    plt.plot(x_0, x_1, linewidth=3, c="brown")

    plt.scatter(*svm_classifier.support_vectors_.T, s=250, zorder=1, c="lime")

    n_betrag = np.linalg.norm(n)
    n_norm = n / n_betrag
    margin_breite = 2 / n_betrag
    margin_vektor = n_norm * margin_breite / 2
    plt.plot(x_0 + margin_vektor[0, 0], x_1 + margin_vektor[0, 1], linewidth=2, linestyle="--", c="orange", zorder=0)

    plt.plot(x_0 - margin_vektor[0, 0], x_1 - margin_vektor[0, 1], linewidth=2, linestyle="--", c="orange", zorder=0)


def plot_confusion_matrix(cm, class_names):
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    ticks = np.array(range(len(class_names)))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    for (pos_y, pos_x), value in np.ndenumerate(cm):
        plt.text(pos_x, pos_y, value, ha='center', va='center', size=12)

    plt.xlabel('Klassifikatorvorhersage')
    plt.ylabel('Referenzstandard')
    plt.show()


def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    layer1 = tf.keras.layers.Dense(256, kernel_size=(3, 3), activation='relu')(inputs)
    # , kernel_regularizer=regularizers.l1(0.01)
    dropout1 = tf.keras.layers.Dropout(0.1)(layer1)

    layer2 = tf.keras.layers.Dense(256, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.1)(layer2)
    layer3 = tf.keras.layers.Dense(128, activation='relu')(dropout2)
    dropout3 = tf.keras.layers.Dropout(0.2)(layer3)
    output = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(dropout3)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="mlp_model")
    print(model.summary())

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def start_training_mlp():
    set_random_seed()  # set seed for reproducibility

    setup_data_generation()


if __name__ == "__main__":
    start_training_mlp()
