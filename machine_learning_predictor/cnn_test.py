#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys
import random
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint


results_folder = "ml_results"
images_train = np.load(os.path.join(results_folder, 'train_images.npy'), allow_pickle=True)  # TODO don't use pickle!
labels_train = np.load(os.path.join(results_folder, 'train_labels.npy'), allow_pickle=True)
paths_train = np.load(os.path.join(results_folder, 'train_paths.npy'), allow_pickle=True)

images_test = np.load(os.path.join(results_folder, 'test_images.npy'), allow_pickle=True)
labels_test = np.load(os.path.join(results_folder, 'test_labels.npy'), allow_pickle=True)
paths_test = np.load(os.path.join(results_folder, 'test_paths.npy'), allow_pickle=True)

MODEL_NAME = 'Cognitive-Load-CNN-Model'
model_path = os.path.join(results_folder, MODEL_NAME)

checkpoint_folder = "./checkpoints"
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)

checkpoint_path = os.path.join(results_folder, checkpoint_folder,
                               "checkpoint-improvement-{epoch:02d}-{categorical_accuracy:.3f}.h5")

label_encoding = {"hard": [0, 0, 1], "medium": [0, 1, 0], "easy": [1, 0, 0]}


def random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def test_model():

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
            for k, v in label_encoding.items():
                if all(v == actual_label_vector):
                    correct_label = k
                    break

            for k, v in label_encoding.items():
                if v[highest_index]:  # TODO highest_index - 1 ?
                    plt.title(f"{probability_vector[highest_index] * 100:.0f}% {k} ({correct_label})")
                    break

        plt.savefig(os.path.join(results_folder, 'result_classification.png'))

    if os.path.exists(model_path):
        loaded_model = keras.models.load_model(model_path)
        print("Model successfully loaded")

        prediction = loaded_model.predict(images_test)
        # print(f"Prediction result: {prediction}")
        show_imgs_with_prediction(paths_test, labels_test, prediction)

        test_loss, test_acc = loaded_model.evaluate(images_test, labels_test, verbose=1)
        print("Test loss: ", test_loss)
        print("Test accuracy: ", test_acc * 100)
    else:
        sys.stderr.write("No ml model found!")


def train_model():
    print("Len train data:", len(images_train))
    print("Len test data:", len(images_test))

    # TODO show some images in the train set first
    # show_samples(images_train)

    random_seed()

    model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=images_train.shape[1:]),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.3),

            # keras.layers.Conv2D(64, (3, 3), activation='relu'),
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            # units in the last layer should be a power of two (e.g. 64, 128, 512, 1024)
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dropout(0.5),

            # units=3 as we have 3 classes -> we need a vector that looks like this: [0.2, 0.5, 0.3]
            keras.layers.Dense(units=3, activation="softmax")  # softmax for multi-class classification, see
            # https://medium.com/deep-learning-with-keras/how-to-solve-classification-problems-in-deep-learning-with-tensorflow-keras-6e39c5b09501
        ]
    )

    model.summary()
    # optimizer = SGD(lr=0.01, momentum=0.9)
    # oters optimizers like "rmsprop" or "adamax" ?

    # for the choice of last layer, activation and loss functions see
    # https://medium.com/deep-learning-with-keras/which-activation-loss-functions-in-multi-class-clasification-4cd599e4e61f
    # metrics=['categorical_accuracy'])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])

    EPOCHS = 25
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.25

    # save checkpoints
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1, mode="max",
                                          save_best_only=True)
    # TODO load checkpoint with:
    # model.load_weights(checkpoint_path)

    # history = model.fit(images_train, labels_train, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
    #                     verbose=1, epochs=EPOCHS, callbacks=[checkpoint_callback])
    history = model.fit(images_train, labels_train, batch_size=BATCH_SIZE, validation_data=(images_test, labels_test),
                        verbose=1, epochs=EPOCHS, callbacks=[checkpoint_callback, keras.callbacks.ReduceLROnPlateau()])
    print(history.history)

    model.save(model_path)

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    show_result_plot(acc, val_acc, loss, val_loss, EPOCHS)


def show_result_plot(accuracy, val_accuracy, loss, val_loss, epochs):
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # save plot to file and show in a new window
    plt.savefig(os.path.join(results_folder, 'train_history.png'))
    plt.show()


if __name__ == "__main__":
    train_model()
    # test_model()
