import itertools
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import metrics
from post_processing.extract_downloaded_data import get_smallest_fps
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import RANDOM_SEED, results_folder, NEW_IMAGE_SIZE, \
    DEFAULT_TRAIN_SPLIT, NUMBER_OF_CLASSES


def set_random_seed(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def get_suitable_sample_size(category_size):
    # use a divisor of the amount of images per difficulty category for a participant
    # -> this way their won't be any overlap of label categories or participants per sample!
    sample_size = 1
    min_sample_size = 30
    for i in range(min_sample_size, 201):
        if category_size % i == 0:
            sample_size = i
            break

    """
    # fps = get_smallest_fps()
    sample_time_span = 6  # 6 seconds as in the Fridman Paper: "Cognitive Load Estimation in the Wild"
    sample_size = round(fps * sample_time_span)  # the number of images we take as one sample
    """
    print("[INFO] Using sample size: ", sample_size)
    return sample_size


def split_train_test(participant_list, train_ratio=DEFAULT_TRAIN_SPLIT):
    random.shuffle(participant_list)

    train_split = int(len(participant_list) * train_ratio)
    train_participants = participant_list[:train_split]
    test_participants = participant_list[train_split:]
    print(f"{len(train_participants)} participants used for training: {train_participants}")
    print(f"{len(test_participants)} participants used for validation: {test_participants}")

    return train_participants, test_participants


def configure_dataset_for_performance(ds, filename=None):
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


def crop_center_square(frame):
    # taken from https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
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


def show_image(image, label=None):
    plt.figure()
    plt.imshow(image)
    if label is not None:
        plt.title(label)
    plt.axis('off')
    plt.show()


def show_result_plot(train_history, metric="categorical_accuracy", output_folder=results_folder,
                     output_name="train_history.png", show=True):

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

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    # save plot to file and show in a new window
    plt.savefig(os.path.join(output_folder, output_name))
    if show:
        plt.show()


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


def calculate_prediction_results(true_labels, predicted_labels):
    label_names = DifficultyLevels.values()
    print(f"\nAccuracy score on test data: {metrics.accuracy_score(true_labels, predicted_labels) * 100:.2f} %")
    print(f"Balanced accuracy score on test data: {metrics.balanced_accuracy_score(true_labels, predicted_labels):.2f}")
    print(f"Precision score on test data: "
          f"{metrics.precision_score(true_labels, predicted_labels, average='weighted'):.2f}")
    print(f"F1 score on test data: {metrics.f1_score(true_labels, predicted_labels, average='weighted'):.2f}")
    try:
        print(f"\nClassification Report:\n"
              f"{metrics.classification_report(true_labels, predicted_labels, target_names=label_names)}")
    except Exception as e:
        sys.stderr.write(f"Failed to compute classification report: {e}")

    # compute and show the confusion matrix
    try:
        conf_matrix = metrics.confusion_matrix(predicted_labels, true_labels, normalize="all")
        print(f"Confusion Matrix:\n{conf_matrix}")
        plot_confusion_matrix(conf_matrix, label_names)
    except Exception as e:
        sys.stderr.write(f"Failed to compute confusion matrix: {e}")


def load_saved_model(model_name):
    model_path = os.path.join(results_folder, model_name)

    if os.path.exists(model_path):
        loaded_model = tf.keras.models.load_model(model_path)
        # tf.keras.utils.plot_model(loaded_model, "loaded_model_graph.png", show_shapes=True)
        print("Model successfully loaded")
        return loaded_model
    else:
        sys.stderr.write("No saved model found!")
        return None


def save_prediction_as_image(batch, sequence_number, actual_label, predicted_label):
    sequence = batch[sequence_number]
    sequence_len = int(sequence.shape[1] / sequence.shape[0])   # calculate the sequence length based on the shape
    img_height, img_width = NEW_IMAGE_SIZE

    plt.figure(figsize=(10, 10))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sequence[:, 0:sequence_len * img_width, :])
    plt.ylabel(DifficultyLevels.get_label_for_encoding(actual_label))
    plt.title(f"Predicted label: {predicted_label}")

    # plt.savefig(os.path.join(results_folder, f"mixed_data_prediction_result_{sequence_number}_{time.time()}.png"))
    plt.savefig(os.path.join(results_folder, f"mixed_data_prediction_result_{sequence_number}.png"))


def get_label_name_for_index_pos(index_pos):
    mask_array = np.zeros(NUMBER_OF_CLASSES, dtype=int)  # creates this: [0 0 0]
    mask_array[index_pos] = 1  # if index_pos was 0: [1 0 0]
    return DifficultyLevels.get_label_for_encoding(mask_array)


def predict_new_data(model, img_batch, eye_data_batch, correct_labels):
    predictions = model.predict([img_batch, eye_data_batch])

    for i, (prediction, correct_label) in enumerate(zip(predictions, correct_labels)):
        score = tf.nn.softmax(prediction)
        print(f"\nPrediction for sequence {i}: {prediction}\nScore: {score})")
        index = np.argmax(score)
        predicted_label = get_label_name_for_index_pos(index)
        print(f"Correct label is  \"{DifficultyLevels.get_label_for_encoding(correct_label)}\"")
        print(f"Predicted label was \"{predicted_label}\" with a confidence of {100 * score[index]:.2f} %")
        # save_prediction_as_image(img_batch, i, correct_label, predicted_label)

    return predictions
