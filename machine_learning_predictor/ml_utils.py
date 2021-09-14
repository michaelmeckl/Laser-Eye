import os
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import RANDOM_SEED, results_folder, NEW_IMAGE_SIZE


def set_random_seed(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    # crop image first to prevent distortions when resizing
    cropped_frame = frame[start_y: start_y + min_dim, start_x: start_x + min_dim]
    resized_img = tf.image.resize_with_pad(cropped_frame,
                                           target_height=NEW_IMAGE_SIZE[0],
                                           target_width=NEW_IMAGE_SIZE[1])
    return resized_img


def show_sample_images(images, labels, num_images=25):
    row_size = int(np.sqrt(num_images))
    fig_size = row_size * 2

    plt.figure(figsize=(fig_size, fig_size))
    for i in range(num_images):
        ax = plt.subplot(row_size, row_size, i + 1)
        plt.xticks([])
        plt.yticks([])

        # sample_image = images[i, :, :]
        sample_image = images[i]
        plt.imshow(sample_image)
        label = DifficultyLevels.get_label_for_encoding(labels[i])
        plt.xlabel(label)
    plt.show()


def show_image(image, label=None):
    plt.figure()
    plt.imshow(image)
    if label is not None:
        plt.title(label)
    plt.axis('off')
    plt.show()


def show_result_plot(train_history, metric="categorical_accuracy", output_folder=results_folder,
                     output_name="train_history.png"):

    acc = train_history.history[f"{metric}"]
    val_acc = train_history.history[f"val_{metric}"]
    loss = train_history.history["loss"]
    val_loss = train_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # save plot to file and show in a new window
    plt.savefig(os.path.join(output_folder, output_name))
    plt.show()
