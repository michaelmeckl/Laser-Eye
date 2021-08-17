#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import results_folder, NUMBER_OF_CLASSES
from machine_learning_predictor.ml_utils import set_random_seed, show_image, show_result_plot, show_sample_images
from machine_learning_predictor.preprocess_image_data import NEW_IMAGE_SIZE


def show_imgs_with_prediction(image_paths, actual_labels, probabilities):
    plt.figure(figsize=(10, 10))
    num_images = 25
    # image_slice = random.sample(image_paths, num_images)

    for n in range(num_images):
        ax = plt.subplot(5, 5, n + 1)
        img = mpimg.imread(image_paths[n])
        plt.imshow(img)
        plt.axis('off')

        probability_vector = probabilities[n]
        highest_index = np.argmax(probability_vector)
        print(f"Probability Vector: {probability_vector}, highest index: {highest_index}")

        actual_label_vector = actual_labels[n]
        correct_label = None
        for label in DifficultyLevels.values():
            label_vector = DifficultyLevels.get_one_hot_encoding(label)
            if all(label_vector == actual_label_vector):
                correct_label = label
                break

        for label in DifficultyLevels.values():
            label_vector = DifficultyLevels.get_one_hot_encoding(label)
            if label_vector[highest_index]:
                plt.title(f"{probability_vector[highest_index] * 100:.0f}% {label} ({correct_label})")
                break

    plt.savefig(os.path.join(results_folder, 'result_classification.png'))


def test_model(test_data: tuple, model_path, checkpoint_folder_name):
    images_test, labels_test, paths_test = test_data

    if os.path.exists(model_path):
        loaded_model = keras.models.load_model(model_path)
        print("Model successfully loaded")

        prediction = loaded_model.predict(images_test)
        # print(f"Prediction result: {prediction}")
        show_imgs_with_prediction(paths_test, labels_test, prediction)

        test_loss, test_acc = loaded_model.evaluate(images_test, labels_test, verbose=1)
        print("Test accuracy: ", test_acc * 100)

        # load latest (i.e. the best) checkpoint
        loaded_model = keras.models.load_model(model_path)  # re-create the model first!
        checkpoint_folder = os.path.join(results_folder, checkpoint_folder_name)
        latest = tf.train.latest_checkpoint(checkpoint_folder)
        loaded_model.load_weights(latest)

        # and re-evaluate the model
        loss, acc = loaded_model.evaluate(images_test, labels_test, verbose=1)
        print(f"Accuracy with restored model weights: {100 * acc:5.2f}%")
    else:
        sys.stderr.write("No saved model found!")


def build_model_sequential(input_shape, num_classes=NUMBER_OF_CLASSES):
    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            # units in the last layer should be a power of two (e.g. 64, 128, 512, 1024)
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dropout(0.5),

            # units=3 as we have 3 classes -> we need a vector that looks like this: [0.2, 0.5, 0.3]
            keras.layers.Dense(units=num_classes, activation="softmax")  # softmax for multi-class classification, see
            # https://medium.com/deep-learning-with-keras/how-to-solve-classification-problems-in-deep-learning-with-tensorflow-keras-6e39c5b09501
        ]
    )

    model.summary()
    # optimizer = SGD(lr=0.01, momentum=0.9)
    # other optimizers like "rmsprop" or "adamax" ?

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model


def train_model(train_data: tuple, test_data: tuple, model_path, checkpoint_path):
    images_train, labels_train, paths_train = train_data
    images_test, labels_test, paths_test = test_data

    print(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Len train data:", len(images_train))
    print("Len test data:", len(images_test))

    # show some images in the train set first
    # show_sample_images(images_train, labels_train)
    set_random_seed()

    # TODO reshape input to feed 10 images as a sequence; not working correctly atm
    """
    print("Label Shape:", labels_train.shape)
    label_dataset = labels_train[0::10]  # take every 10th element
    print("Label Shape new:", label_dataset.shape)

    print("Img Shape:", images_train.shape)
    print("Img Shape[1:]:", images_train.shape[1:])

    train_dataset = images_train.reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 10)  # TODO not like this! -> 10*10 matrix
    print("train Shape:", train_dataset.shape)
    val_dataset = images_test.reshape(-1, NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 10)
    print("val Shape:", val_dataset.shape)

    test_image = images_train[0, :, :]
    test_image = cv2.resize(test_image, None, fx=5, fy=5)
    cv2.imshow("test", test_image)

    test_image = train_dataset[0, :, :, 1]
    test_image = cv2.resize(test_image, None, fx=5, fy=5)
    show_image(test_image)

    val_label_dataset = labels_test[0::10]  # take every 10th element
    print("Val Label Shape:", val_label_dataset.shape)

    print(train_dataset.shape)
    print(train_dataset.shape[1:])
    """

    # image_shape = images_train.shape[1:]
    image_shape = (NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 1)
    model = build_model_sequential(input_shape=image_shape)

    EPOCHS = 12
    BATCH_SIZE = 10
    VALIDATION_SPLIT = 0.25

    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1, mode="max",
                                          save_best_only=True, save_weights_only=True)
    lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                                    verbose=1)

    # history = model.fit(images_train, labels_train, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
    #                     verbose=1, epochs=EPOCHS, callbacks=[checkpoint_callback, lr_callback])

    history = model.fit(images_train, labels_train, batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS,
                        validation_data=(images_test, labels_test), callbacks=[checkpoint_callback, lr_callback])
    print(history.history)
    model.save(model_path)

    show_result_plot(history, EPOCHS, metric="categorical_accuracy", output_folder=results_folder)

    # TODO try cross validation:
    """
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    # estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, train, labels, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    """


def main(train=True, test=False):
    # TODO don't use pickle
    images_train = np.load(os.path.join(results_folder, 'train_images.npy'), allow_pickle=False)
    labels_train = np.load(os.path.join(results_folder, 'train_labels.npy'), allow_pickle=False)
    paths_train = np.load(os.path.join(results_folder, 'train_paths.npy'), allow_pickle=False)

    images_test = np.load(os.path.join(results_folder, 'test_images.npy'), allow_pickle=False)
    labels_test = np.load(os.path.join(results_folder, 'test_labels.npy'), allow_pickle=False)
    paths_test = np.load(os.path.join(results_folder, 'test_paths.npy'), allow_pickle=False)

    MODEL_NAME = 'Cognitive-Load-CNN-Model.h5'
    model_save_location = os.path.join(results_folder, MODEL_NAME)

    checkpoint_folder = "checkpoints"
    checkpoint_path = os.path.join(results_folder, checkpoint_folder,
                                   "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")

    if train:
        train_model((images_train, labels_train, paths_train), (images_test, labels_test, paths_test),
                    model_save_location, checkpoint_path)

    if test:
        # TODO should not be the ones used for validation in train_model() !
        test_model((images_test, labels_test, paths_test), model_save_location, checkpoint_folder)


if __name__ == "__main__":
    main(train=True, test=True)
