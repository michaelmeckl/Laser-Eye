#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
from machine_learning_predictor.machine_learning_constants import NUMBER_OF_CLASSES, data_folder_path, RANDOM_SEED

# for reproducibility set this BEFORE importing tensorflow: see
# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

import random
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from machine_learning_predictor.ml_utils import set_random_seed, split_train_test, show_result_plot, \
    get_suitable_sample_size
from post_processing.post_processing_constants import post_processing_log_folder
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# make pandas and numpy output more pretty and show complete dataframe
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 25)
np.set_printoptions(linewidth=500)


def merge_participant_eye_tracking_logs(participant_list, is_train_data):
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

    # add label encoding to the dataframes
    blink_dataframe['difficulty_level_number'] = blink_dataframe['difficulty_level']
    blink_dataframe.loc[blink_dataframe['difficulty_level'] == 'hard', 'difficulty_level_number'] = 2
    blink_dataframe.loc[blink_dataframe['difficulty_level'] == 'medium', 'difficulty_level_number'] = 1
    blink_dataframe.loc[blink_dataframe['difficulty_level'] == 'easy', 'difficulty_level_number'] = 0

    eye_log_dataframe['difficulty_level_number'] = eye_log_dataframe['difficulty_level']
    eye_log_dataframe.loc[eye_log_dataframe['difficulty_level'] == 'hard', 'difficulty_level_number'] = 2
    eye_log_dataframe.loc[eye_log_dataframe['difficulty_level'] == 'medium', 'difficulty_level_number'] = 1
    eye_log_dataframe.loc[eye_log_dataframe['difficulty_level'] == 'easy', 'difficulty_level_number'] = 0

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
    # convert object data types back to floats
    string_dtypes = ['ROLL', 'PITCH', 'YAW', 'LEFT_PUPIL_POS_X', 'LEFT_PUPIL_POS_Y', 'RIGHT_PUPIL_POS_X',
                     'RIGHT_PUPIL_POS_Y', 'LEFT_EYE_CENTER_X', 'LEFT_EYE_CENTER_Y', 'RIGHT_EYE_CENTER_X',
                     'RIGHT_EYE_CENTER_Y']
    eye_log_dataframe_ordered[string_dtypes] = eye_log_dataframe_ordered[string_dtypes].astype(float)
    # print(eye_log_dataframe_ordered.dtypes)

    # save as csv files
    eye_log_dataframe_ordered.to_csv(f"eye_log_dataframe_ordered_{'train' if is_train_data else 'val'}.csv",
                                     index=False)
    blink_dataframe_shuffled.to_csv(f"blink_dataframe_shuffled_{'train' if is_train_data else 'val'}.csv", index=False)

    return blink_dataframe_shuffled, eye_log_dataframe_ordered


def load_pupil_movement_data(train_participants):
    pupil_movement_dataframe = pd.DataFrame()
    pupil_movement_path = "pupil_movement_data"

    for csv_file in os.listdir(pupil_movement_path):
        csv_df = pd.read_csv(os.path.join(pupil_movement_path, csv_file))
        pupil_movement_dataframe = pd.concat([pupil_movement_dataframe, csv_df])

    pupil_movement_dataframe['difficulty_level_number'] = pupil_movement_dataframe['difficulty']
    pupil_movement_dataframe.loc[pupil_movement_dataframe['difficulty'] == 'hard', 'difficulty_level_number'] = 2
    pupil_movement_dataframe.loc[pupil_movement_dataframe['difficulty'] == 'medium', 'difficulty_level_number'] = 1
    pupil_movement_dataframe.loc[pupil_movement_dataframe['difficulty'] == 'easy', 'difficulty_level_number'] = 0

    # add one-hot-encoding to the dataframe
    new_cols = pd.get_dummies(pupil_movement_dataframe['difficulty'], prefix='difficulty')
    pupil_movement_dataframe = pd.concat([pupil_movement_dataframe, new_cols], axis=1)

    pupil_movement_dataframe = pupil_movement_dataframe.reset_index(drop=True)

    pupil_movement_dataframe[["difficulty_level_number"]] = pupil_movement_dataframe[["difficulty_level_number"]].astype(int)

    train_pupil_movement = pupil_movement_dataframe[pupil_movement_dataframe.participant.isin(train_participants)]
    val_pupil_movement = pupil_movement_dataframe[~pupil_movement_dataframe.participant.isin(train_participants)]

    train_pupil_movement.to_csv("train_pupil_movement.csv", index=False)
    val_pupil_movement.to_csv("val_pupil_movement.csv", index=False)

    return train_pupil_movement, val_pupil_movement


def setup_data_generation(all_participants):
    # TODO cross validation: split into random participants
    # train_test_split(all_participants, test_size=0.2, random_state=RANDOM_SEED)
    # score = cross_val_score(pipe, X_all, y_all, cv=3, scoring='accuracy').mean()
    # print("Score: {:.2%}".format(score))

    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    # evaluate the pipeline using cross validation and calculate MAE
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # convert MAE scores to positive values
    scores = absolute(scores)
    """

    # split into train and test data
    train_participants, test_participants = split_train_test(all_participants)

    # TODO always needs to be re-generated if seed or participants are changed!
    if os.path.exists("eye_log_dataframe_ordered_train.csv"):
        # load data from csv
        eye_log_train_data = pd.read_csv("eye_log_dataframe_ordered_train.csv")
        eye_log_val_data = pd.read_csv("eye_log_dataframe_ordered_val.csv")
        blink_train_data = pd.read_csv("blink_dataframe_shuffled_train.csv")
        blink_val_data = pd.read_csv("blink_dataframe_shuffled_val.csv")

        pupil_move_train = pd.read_csv("train_pupil_movement.csv")
        pupil_move_val = pd.read_csv("val_pupil_movement.csv")
    else:
        # generate the data new
        blink_train_data, eye_log_train_data = merge_participant_eye_tracking_logs(train_participants,
                                                                                   is_train_data=True)
        blink_val_data, eye_log_val_data = merge_participant_eye_tracking_logs(test_participants, is_train_data=False)
        pupil_move_train, pupil_move_val = load_pupil_movement_data(train_participants)

    for difficulty_level in pupil_move_train.difficulty.unique():
        difficulty_level_df = pupil_move_train[pupil_move_train.difficulty == difficulty_level]
        print(f"Found {len(difficulty_level_df)} train samples for category \"{difficulty_level}\".\n")

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
    batch_size = 50

    sample_size = get_suitable_sample_size(difficulty_category_size)
    print(f"Sample size: {sample_size}")

    column_trans = ColumnTransformer(
        [
            ('scale', StandardScaler(), make_column_selector(dtype_include=np.number)),
            ('difficulty_category', OneHotEncoder(dtype='int'), ['difficulty_level']),  # one-hot-encoder needs a list
        ], remainder='passthrough'
    )
    # column_trans.fit(eye_log_train_data)
    # test = column_trans.transform(eye_log_train_data)

    # TODO
    # transformer = ColumnTransformer(transformers=[('one_hot', OneHotEncoder(), ["difficulty_level"])])
    # pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    # pipeline.fit(train_X, train_y)
    # pipeline.predict(X_val)

    # print("\nCorrelations_Pupil_Move_Train:\n", pupil_move_train.corr())

    # print("\nCorrelations_Eye_Log_Train:\n", eye_log_train_data.corr())
    # print("\nCorrelations_Eye_Log_Val:\n", eye_log_val_data.corr())

    # print("\nCorrelations_Blink_Log_Train:\n", blink_train_data.corr())
    # print("\nCorrelations_Blink_Log_Val:\n", blink_val_data.corr())

    # print("\nEye Log Train descriptive:\n", eye_log_train_data.describe())

    pupil_movement_feature_vectors = {
        "avg_dist": {"average_pupil_movement_distance"},
        "all": ["left_pupil_movement_x", "left_pupil_movement_y", "right_pupil_movement_x", "right_pupil_movement_y",
                "average_pupil_movement_x", "average_pupil_movement_y", "average_pupil_movement_distance",
                "time_difference"],
    }

    blink_feature_vectors = {
        "best_correlation": ["total_blinks", "avg_blinks_per_minute"],
        "slight_correlation": ["min_blink_duration_in_ms", "max_blink_duration_in_ms", "avg_blink_duration_in_ms"],
        "all": ["total_blinks", "avg_blinks_per_minute", "min_blink_duration_in_ms", "max_blink_duration_in_ms",
                "avg_blink_duration_in_ms"],
    }

    eye_log_feature_vectors = {
        "best_correlation": ["ROLL", "PITCH"],  # , "YAW"],

        "slight_correlation": ["LEFT_PUPIL_POS_X", "LEFT_PUPIL_POS_Y", "RIGHT_PUPIL_POS_X", "RIGHT_PUPIL_POS_Y",
                               "LEFT_EYE_CENTER_X", "LEFT_EYE_CENTER_Y", "RIGHT_EYE_CENTER_X", "RIGHT_EYE_CENTER_Y"],

        "slight_correlation_alternative": ["LEFT_EYE_WIDTH", "RIGHT_EYE_WIDTH", "LEFT_EYE_HEIGHT", "RIGHT_EYE_HEIGHT"],

        "correlation_all": ["ROLL", "PITCH", "YAW", "LEFT_PUPIL_POS_X", "LEFT_PUPIL_POS_Y",
                            "RIGHT_PUPIL_POS_X", "RIGHT_PUPIL_POS_Y", "LEFT_EYE_CENTER_X",
                            "LEFT_EYE_CENTER_Y", "RIGHT_EYE_CENTER_X", "RIGHT_EYE_CENTER_Y"],

        "all": ["LEFT_PUPIL_POS_X", "LEFT_PUPIL_POS_Y", "RIGHT_PUPIL_POS_X", "RIGHT_PUPIL_POS_Y",
                "LEFT_EYE_CENTER_X", "LEFT_EYE_CENTER_Y", "RIGHT_EYE_CENTER_X", "RIGHT_EYE_CENTER_Y",
                "LEFT_EYE_WIDTH", "RIGHT_EYE_WIDTH", "LEFT_EYE_HEIGHT", "RIGHT_EYE_HEIGHT", "ROLL", "PITCH", "YAW"],
    }

    # First results:
    # all blink features mit svm 50 %
    # best correlation blink features mit perceptron 50 %
    # all blink features mit mlp (one-hot) und batch_size auskommentiert über 70 epochs 58 %

    """
    X_train = eye_log_train_data[eye_log_feature_vectors["all"]].values
    X_val = eye_log_val_data[eye_log_feature_vectors["all"]].values
    y_train = eye_log_train_data["difficulty_level_number"].values
    y_val = eye_log_val_data["difficulty_level_number"].values
    y_train_one_hot = eye_log_train_data[["difficulty_hard", "difficulty_medium", "difficulty_easy"]].values
    y_val_one_hot = eye_log_val_data[["difficulty_hard", "difficulty_medium", "difficulty_easy"]].values
    """

    """
    X_train = blink_train_data[blink_feature_vectors["all"]].values
    X_val = blink_val_data[blink_feature_vectors["all"]].values
    y_train = blink_train_data["difficulty_level_number"].values
    y_val = blink_val_data["difficulty_level_number"].values
    y_train_one_hot = blink_train_data[["difficulty_hard", "difficulty_medium", "difficulty_easy"]].values
    y_val_one_hot = blink_val_data[["difficulty_hard", "difficulty_medium", "difficulty_easy"]].values
    """

    X_train = pupil_move_train[pupil_movement_feature_vectors["all"]].values
    X_val = pupil_move_val[pupil_movement_feature_vectors["all"]].values
    y_train = pupil_move_train["difficulty_level_number"].values
    y_val = pupil_move_val["difficulty_level_number"].values
    y_train_one_hot = pupil_move_train[["difficulty_hard", "difficulty_medium", "difficulty_easy"]].values
    y_val_one_hot = pupil_move_val[["difficulty_hard", "difficulty_medium", "difficulty_easy"]].values

    print(X_train[:5])

    """
    # show scatterplot for all difficulty levels
    plt.scatter(*X_val[y_val == 2].T, label="hard", c='steelblue')
    plt.scatter(*X_val[y_val == 1].T, label="medium", c='red')
    plt.scatter(*X_val[y_val == 0].T, label="easy", c='orange')
    plt.xlabel("ROLL")
    plt.ylabel("PITCH")
    plt.legend()
    plt.show()
    """

    # TODO use this to get every sample_size-th row of df starting from the first
    # sampled_df = eye_log_train_data[0::sample_size]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_perceptron(X_train, y_train, X_val, y_val)
    # train_svm(X_train, y_train, X_val, y_val)
    # TODO
    # train_mlp(X_train, y_train_one_hot, X_val, y_val_one_hot, batch_size)


def train_perceptron(X_train, y_train, X_val, y_val):
    print("\n Training perceptron:\n")
    perceptron = Perceptron(random_state=RANDOM_SEED, shuffle=True)
    perceptron.fit(X_train, y_train)
    prediction_perceptron = perceptron.predict(X_val)
    print("Prediction Perceptron:\n", prediction_perceptron)
    print(f"Accuracy: {metrics.accuracy_score(y_val, prediction_perceptron) :.2%} correctly classified!")
    print(f"Bal. Accuracy: {metrics.balanced_accuracy_score(y_val, prediction_perceptron):.2%} correctly classified!")

    # class_names = DifficultyLevels.values()
    # cm = confusion_matrix(y_val, prediction_perceptron, normalize="all")
    # print(cm)
    # plot_confusion_matrix(cm, class_names)


def train_svm(X_train, y_train, X_val, y_val):
    print("\n Training svm:\n")
    svc = SVC(kernel="linear", C=10, random_state=RANDOM_SEED, verbose=1)

    svc.fit(X_train, y_train)
    prediction_svm = svc.predict(X_val)
    print("Prediction SVM:\n", prediction_svm)
    print(f"Accuracy: {metrics.accuracy_score(y_val, prediction_svm) :.2%} correctly classified!")
    print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_val, prediction_svm):.2%} correctly classified!")

    print("SVM Score: ", svc.score(X_val, y_val))
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
    print("\nInput_shape: ", input_shape)

    inputs = tf.keras.Input(shape=input_shape)
    layer1 = tf.keras.layers.Dense(512, activation='relu')(inputs)  # , kernel_regularizer=regularizers.l1(0.01)

    layer2 = tf.keras.layers.Dense(256, activation='relu')(layer1)
    layer3 = tf.keras.layers.Dense(256, activation='relu')(layer2)
    layer4 = tf.keras.layers.Dense(32, activation='relu')(layer3)
    dropout3 = tf.keras.layers.Dropout(0.2)(layer4)

    output = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(dropout3)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="mlp_model")
    print(model.summary())

    # TODO 'sparse_categorical_crossentropy' as loss instead
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_mlp(X_train, y_train, X_val, y_val, batch_size):
    model = build_model(X_train.shape[1])

    history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                        # batch_size=batch_size,
                        use_multiprocessing=False,
                        workers=6,
                        epochs=20,
                        verbose=1)

    print(history.history)
    show_result_plot(history, metric="accuracy", output_name="train_history_mlp.png",
                     show=True)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
    print("Validation loss: ", val_loss)
    print("Validation accuracy: ", val_acc * 100)

    # TODO mit kerasClassifier:
    """
    keras_classifier = KerasClassifier(build_classifier, epochs=20, batch_size=256,verbose = 0, l2 = 0.1)
    """


def start_training_mlp():
    set_random_seed()  # set seed for reproducibility

    without_participants = []

    all_participants = os.listdir(data_folder_path)[:18]  # only take 12 or 18 so the counterbalancing works
    # remove some participants for testing
    all_participants = [p for p in all_participants if p not in set(without_participants)]

    setup_data_generation(all_participants)


if __name__ == "__main__":
    start_training_mlp()
