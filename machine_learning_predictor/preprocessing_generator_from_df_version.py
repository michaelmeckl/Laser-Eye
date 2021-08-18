#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import pathlib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from machine_learning_predictor.machine_learning_constants import data_folder_path
from post_processing.post_processing_constants import download_folder


def merge_participant_image_logs(participant_list):
    image_data_frame = pd.DataFrame()
    post_processing_folder_path = pathlib.Path(__file__).parent.parent / "post_processing"

    for participant in participant_list:
        images_label_log = post_processing_folder_path / download_folder / participant / "labeled_images.csv"
        labeled_images_df = pd.read_csv(images_label_log)
        labeled_images_df = labeled_images_df.sample(n=450)  # TODO for testing take only 450 rows randomly
        image_data_frame = pd.concat([image_data_frame, labeled_images_df])

    # add the index numbers as own column (reset the index first as the concatenate above creates duplicate indexes)
    image_data_frame_numbered = image_data_frame.reset_index(drop=True)
    image_data_frame_numbered["index"] = image_data_frame_numbered.index

    # shuffle the dataframe rows but maintain the row content,
    # see https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # image_data_frame_numbered = image_data_frame_numbered.sample(frac=1)
    # TODO don't shuffle df, only choose indices in generator at random!

    return image_data_frame_numbered


def get_suitable_batch_size(dataset_len: int, test_data=False):
    # Find a suitable batch size that is a divisor of the number of images in the data (1 would obviously work but
    # one batch would be the whole dataset then). For the training generator finding a perfect divisor isn't as
    # important as for testing, so we simply use 32 as default if no other divisor is found.
    batch_size = 1 if test_data else 32
    for i in range(10, 101):  # start from 10 so the batches won't be too large (which would slow it down quite a bit)
        if dataset_len % i == 0:
            batch_size = i
            break

    return batch_size


def split_train_test(participant_list, train_ratio=0.8):
    random.shuffle(participant_list)

    train_split = int(len(participant_list) * train_ratio)
    train_participants = participant_list[:train_split]
    test_participants = participant_list[train_split:]
    print("Number of participants used for training: ", len(train_participants))
    print("Number of participants used for testing: ", len(test_participants))

    return train_participants, test_participants


def build_model_sequential(input_shape, num_classes=3):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            # units in the last layer should be a power of two (e.g. 64, 128, 512, 1024)
            tf.keras.layers.Dense(units=1024, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            # units=3 as we have 3 classes -> we need a vector that looks like this: [0.2, 0.5, 0.3]
            tf.keras.layers.Dense(units=num_classes, activation="softmax")  # softmax for multi-class classification, see
            # https://medium.com/deep-learning-with-keras/how-to-solve-classification-problems-in-deep-learning-with-tensorflow-keras-6e39c5b09501
        ]
    )

    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def preprocess_image_data():
    random_seed = 42
    random.seed(random_seed)  # set seed for reproducibility

    new_image_size = (150, 150)

    all_participants = os.listdir(data_folder_path)[:12]  # TODO only take 12 so the counterbalancing works
    train_participants, test_participants = split_train_test(all_participants)

    train_data = merge_participant_image_logs(train_participants)
    test_data = merge_participant_image_logs(test_participants)

    train_batch_size = get_suitable_batch_size(len(train_data), test_data=False)
    test_batch_size = get_suitable_batch_size(len(test_data), test_data=True)
    print(f"Train batch size: {train_batch_size} (Data len: {len(train_data)})")
    print(f"Test batch size: {test_batch_size} (Data len: {len(test_data)})")

    images_base_path = pathlib.Path(__file__).parent.parent / "post_processing"
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)  #, validation_split=0.25)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=images_base_path,
        x_col="image_path",
        y_col="load_level",
        # subset="training",
        batch_size=train_batch_size,
        seed=random_seed,
        shuffle=False,  # todo shuffle=True ruins the time dim ...
        validate_filenames=False,
        color_mode="rgb",
        class_mode="categorical",
        target_size=new_image_size)

    """
    valid_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=images_base_path,
        x_col="image_path",
        y_col="load_level",
        subset="validation",
        batch_size=train_batch_size,
        seed=random_seed,
        shuffle=True,
        validate_filenames=False,
        color_mode="rgb",
        class_mode="categorical",
        target_size=new_image_size)
    """

    # in this case we do have the labels!
    """
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=images_base_path,
        x_col="image_path",
        y_col=None,  # use None to return only the images without the labels
        class_mode=None,
        batch_size=test_batch_size,
        seed=random_seed,
        shuffle=False,
        validate_filenames=False,
        color_mode="rgb",
        target_size=new_image_size)
    """
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=images_base_path,
        x_col="image_path",
        y_col="load_level",
        class_mode="categorical",
        batch_size=test_batch_size,
        seed=random_seed,
        shuffle=False,
        validate_filenames=False,
        color_mode="rgb",
        target_size=new_image_size)

    """
    X, y = train_generator[0]
    plt.figure()
    plt.imshow(X[1])
    plt.axis('off')
    plt.title(f"Label: {y[1]}")
    plt.show()

    X = test_generator[0]
    plt.figure()
    plt.imshow(X[1])
    plt.axis('off')
    plt.title("Test image")
    plt.show()
    """

    channels = 3  # if 'rgb' else 1
    model = build_model_sequential(input_shape=(*new_image_size, channels), num_classes=3)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.sample_size
    # STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.sample_size

    # history = model.fit_generator(generator=train_generator,
    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=test_generator,
                        validation_steps=STEP_SIZE_TEST,
                        epochs=20)
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(20)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    test_loss, test_acc = model.evaluate(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    print("test loss: ", test_loss)
    print("test accuracy: ", test_acc * 100)

    test_true = False
    if test_true:
        test_generator.reset()
        pred = model.predict(test_generator,
                             steps=STEP_SIZE_TEST,
                             verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        labels = (train_generator.class_indices)
        labels = dict((v, k) for k, v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        filenames = test_generator.filenames
        results = pd.DataFrame({"Filename": filenames,
                                "Predictions": predictions})
        results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    preprocess_image_data()
