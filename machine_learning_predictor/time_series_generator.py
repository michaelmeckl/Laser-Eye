import os
import random
import sys
import numpy as np
import tensorflow as tf
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import NEW_IMAGE_SIZE, NUMBER_OF_CLASSES
from machine_learning_predictor.ml_utils import crop_center_square


class CustomImageDataGenerator(tf.keras.utils.Sequence):
    """
    Structure based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, data_frame, x_col_name, y_col_name, sequence_length, batch_size, num_classes=NUMBER_OF_CLASSES,
                 images_base_path=".", use_grayscale=False, is_train_set=True):

        self.df = data_frame.copy()
        self.X_col = x_col_name
        self.y_col = y_col_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.n_classes = num_classes
        self.images_base_path = images_base_path
        self.use_grayscale = use_grayscale
        self.is_train = is_train_set

        self.n = len(self.df)

        num_channels = 1 if self.use_grayscale else 3
        self.output_size = (*NEW_IMAGE_SIZE, num_channels)

        # create a random order for the samples
        self.indices_list = self.generate_random_index_list()

    def generate_random_index_list(self):
        sample_indices = []
        for i in range(0, self.n, self.sequence_length):
            sample_indices.append(i)

        random.shuffle(sample_indices)
        return sample_indices

    def __len__(self):
        length = self.n // (self.sequence_length * self.batch_size)
        return length

    def on_epoch_end(self):
        # self.indices_list = self.generate_random_index_list()  # each epoch we generate a new indices order
        random.shuffle(self.indices_list)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, self.sequence_length, *self.output_size), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes))

        # start from the current batch and take the next 'batch_size' indices
        current_index = index * self.batch_size
        indices = self.indices_list[current_index:current_index + self.batch_size]

        for i, start_index in enumerate(indices):
            sample_rows = self.df[start_index:start_index + self.sequence_length]  # get the corresponding df rows
            image_sample, sample_label = self.__get_data(sample_rows)
            X[i, ] = image_sample
            y[i, ] = sample_label

        return X, y

    def __get_data(self, sample):
        image_sample = np.empty((self.sequence_length, *self.output_size), dtype=np.float32)

        # Load and preprocess the images for the current sample
        i = 0
        for idx, row in sample.iterrows():
            img_path = row[self.X_col]
            image_path = os.path.join(self.images_base_path, img_path)
            image_sample[i, ] = self.__scale_and_convert_image(image_path)  # load image and resize and scale it
            i += 1

        label = sample[self.y_col].iloc[0]  # take the label of the first element in the sample
        sample_label = DifficultyLevels.get_one_hot_encoding(label)  # convert string label to one-hot-vector

        return image_sample, sample_label

    def __scale_and_convert_image(self, image_path):
        try:
            color_mode = "grayscale" if self.use_grayscale else "rgb"

            image = tf.keras.preprocessing.image.load_img(image_path, color_mode=color_mode)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            resized_img = crop_center_square(image_arr)

            # normalize pixel values to [0, 1] so the CNN can work with smaller values
            scaled_img = resized_img / 255.0
            return scaled_img

        except Exception as e:
            sys.stderr.write(f"\nError in processing image '{image_path}': {e}")
            return None

    def get_image_shape(self):
        return self.batch_size, self.sequence_length, *self.output_size

    def get_example_batch(self, idx=0):
        # we need to make a copy first so we don't actually change the list by taking an example
        indices_copy = self.indices_list.copy()
        first_sample, labels = self.__getitem__(idx)
        self.indices_list = indices_copy
        return first_sample, labels
