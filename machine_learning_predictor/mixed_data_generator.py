import os
import random
import sys
import numpy as np
import tensorflow as tf
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.feature_extraction.pupil_movement_calculation import PupilMovementCalculation
from machine_learning_predictor.machine_learning_constants import NEW_IMAGE_SIZE, NUMBER_OF_CLASSES
from machine_learning_predictor.ml_utils import crop_center_square


class MixedDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Generator that loads, preprocesses and returns the images, eye log data and their corresponding labels in
    sequences to be used in a mixed-input model.
    Structure and implementation is based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
    The custom Generator inherits from `Sequence` to support multi-threading.
    """

    def __init__(self, img_data_frame, eye_data_frame, x_col_name, y_col_name, sequence_length, batch_size,
                 num_classes=NUMBER_OF_CLASSES, images_base_path=".", use_grayscale=False, is_train_set=True):

        self.image_df = img_data_frame.copy()
        self.eye_df = eye_data_frame.copy()

        self.X_col = x_col_name
        self.y_col = y_col_name
        # self.eye_log_cols = ["ROLL", "PITCH"]   # TODO
        self.eye_log_cols = ["average_pupil_movement_distance", "movement_angle"]

        self.apply_fourier_transform = False
        print("\nFourier Transformation applied: ", self.apply_fourier_transform)
        if self.apply_fourier_transform:
            self.pupil_movement_calculator = PupilMovementCalculation()

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.images_base_path = images_base_path
        self.use_grayscale = use_grayscale
        self.is_train = is_train_set

        num_channels = 1 if self.use_grayscale else 3
        self.original_image_shape = (*NEW_IMAGE_SIZE, num_channels)
        self.stacked_image_shape = (self.original_image_shape[0], (self.sequence_length * self.original_image_shape[1]),
                                    self.original_image_shape[2])
        self.eye_log_shape = (self.sequence_length, len(self.eye_log_cols))

        self.img_n = len(self.image_df)
        self.eye_n = len(self.eye_df)
        print("\nThe following two values must be identical!")
        print(f"Image df len: {self.img_n}")
        print(f"Eye Log df len: {self.eye_n}\n")

        self.indices_list = self.generate_random_index_list()  # create a random order for the samples
        # print(f"Train: {self.is_train}; IndicesList (Len {len(self.indices_list)}): {self.indices_list}")

    def generate_random_index_list(self):
        """
        Iterates over the indices in the dataframe in order and creates batch-sized chunks so consecutive images
        will be fed as a sample to the NN. This way the time-domain can be used as well.
        If we were to randomly choose a new start index on every '__get_item__()' - Method and take "batch-size"
        consecutive images from this index, we would have to remove these from the dataset afterwards so they aren't
        fed more than once to the NN. However, this would also ruin the time-domain as consecutive images from the df
        would not necessarily be correct consecutive images if parts in-between were taken for a previous sample!
        """
        sample_indices = []
        for i in range(0, self.img_n, self.sequence_length):
            sample_indices.append(i)

        random.shuffle(sample_indices)
        return sample_indices

    def __len__(self):
        length = self.img_n // (self.sequence_length * self.batch_size)
        return length

    def on_epoch_end(self):
        # self.indices_list = self.generate_random_index_list()  # each epoch we generate a new indices order
        random.shuffle(self.indices_list)

    def __getitem__(self, index):
        """
        Return a new sample in the form (X, y) where X is a list with two elements (a batch of image samples and the
        batch of numerical data from the eye log) and y the corresponding labels.

        Args:
            index: the number of the current sample from 0 to __len__() - 1
        """
        X_img = np.empty((self.batch_size, self.sequence_length, *self.original_image_shape), dtype=np.float32)
        y_img = np.empty((self.batch_size, self.n_classes))

        X_eye = np.empty((self.batch_size, *self.eye_log_shape))
        y_eye = np.empty((self.batch_size, self.n_classes))

        # start from the current batch and take the next 'batch_size' indices
        current_index = index * self.batch_size
        indices = self.indices_list[current_index:current_index + self.batch_size]

        for i, start_index in enumerate(indices):
            # get the corresponding df rows
            image_sample_rows = self.image_df[start_index:start_index + self.sequence_length]
            image_sample, sample_label = self.__get_image_data(image_sample_rows)
            X_img[i, ] = image_sample
            y_img[i, ] = sample_label

            eye_sample_rows = self.eye_df[start_index:start_index + self.sequence_length]
            eye_sample, eye_sample_label = self.__get_eye_data(eye_sample_rows)
            X_eye[i, ] = eye_sample
            y_eye[i, ] = eye_sample_label

        # move the sequence length dim to the left of the width so the reshape works correctly
        X_img = np.moveaxis(X_img, 1, 2)
        # and reshape into (batch_size, img_height, (sequence_length * img_width), num_channels)
        reshaped_X = X_img.reshape(self.batch_size, *self.stacked_image_shape)

        # for both batches the labels should be identical if preprocessing worked correct
        assert (y_eye == y_img).all(), "Error in mixed data generator: labels of image and eye batch are different!"

        # create mixed output
        generator_output = [reshaped_X, X_eye]
        return generator_output, y_img

    def __get_eye_data(self, sample):
        eye_log_sample = np.empty(self.eye_log_shape)

        i = 0
        for idx, row in sample.iterrows():
            # get all columns from the eye feature dataframe that should be used for training
            eye_feature = row[self.eye_log_cols].to_numpy()  # and convert to a numpy array
            eye_log_sample[i, ] = eye_feature
            i += 1

        if self.apply_fourier_transform:
            eye_log_sample = self.pupil_movement_calculator.calculate_frequencies(eye_log_sample)

        # make sure this is the same order as the order of the one-hot vectors in the DifficultyLevels Enum!
        y_one_hot = sample[["difficulty_easy", "difficulty_medium", "difficulty_hard"]].values
        label = y_one_hot[0]  # take only the first as all should be the same (if sequence generation works correctly)

        return eye_log_sample, label

    def __get_image_data(self, sample):
        image_sample = np.empty((self.sequence_length, *self.original_image_shape), dtype=np.float32)

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
        return self.stacked_image_shape

    def get_eye_log_shape(self):
        return self.eye_log_shape

    def get_example_batch(self, idx=0):
        # we need to make a copy first so we don't actually change the list by taking an example
        indices_copy = self.indices_list.copy()
        batch_data, labels = self.__getitem__(idx)
        first_img_sample, first_eye_sample = batch_data[0], batch_data[1]
        self.indices_list = indices_copy
        return first_img_sample, first_eye_sample, labels
