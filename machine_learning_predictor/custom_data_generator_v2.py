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
    Custom Generator that loads, preprocesses and returns the images and their corresponding labels in batches.
    Structure and implementation based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
    The custom Generator inherits from `Sequence` to support multi-threading.
    """

    def __init__(self, data_frame, x_col_name, y_col_name, sequence_length, batch_size, num_classes=NUMBER_OF_CLASSES,
                 images_base_path=".", use_grayscale=False, is_train_set=True):

        self.df = data_frame.copy()
        self.X_col = x_col_name
        self.y_col = y_col_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.images_base_path = images_base_path
        self.use_grayscale = use_grayscale
        self.is_train = is_train_set

        num_channels = 1 if self.use_grayscale else 3
        self.output_size = (*NEW_IMAGE_SIZE, num_channels)

        # self.new_image_shape = ((self.sequence_length * self.output_size[0]), self.output_size[1],self.output_size[2])
        self.new_image_shape = (self.output_size[0], (self.sequence_length * self.output_size[1]), self.output_size[2])

        self.n = len(self.df)
        self.indices_list = self.generate_random_index_list()  # create a random order for the samples
        # print(f"Train: {self.is_train}; IndicesList (Len {len(self.indices_list)}): {self.indices_list}")

    def generate_random_index_list(self):
        """
        Iterates over the indices in the dataframe in order and creates batch-sized chunks so consecutive images
        will be fed as a sample to the CNN. This way the time-domain can be used as well.
        If we were to randomly choose a new start index on every '__get_item__()' - Method and take "batch-size"
        consecutive images from this index, we would have to remove these from the dataset afterwards so they aren't
        fed more than once to the CNN. However, this would also ruin the time-domain as consecutive images from the df
        would not necessarily be correct consecutive images if parts in-between were taken for a previous sample!
        """
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
        # print(f"\nEpoch finished! Generating new indices list: {self.indices_list}\n", flush=True)

    """
    def __getitem__(self, index):
        X = np.empty((self.batch_size, self.sequence_length, *self.output_size), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes))

        for i, batch in enumerate(range(self.batch_size)):
            if len(self.indices_list) == 0:
                # print(f"\nself.indices_list is empty in __get_item__()!", flush=True)
                continue

            # Get a random index from the list
            start_index = random.choice(self.indices_list)
            # and remove this index from the list so it won't be used more than once per epoch
            self.indices_list.remove(start_index)

            sample_rows = self.df[start_index:start_index + self.sequence_length]  # get the corresponding df rows
            image_sample, sample_label = self.__get_data(sample_rows)
            X[i, ] = image_sample
            y[i, ] = sample_label

        # move the sequence length dim to the left of the width so the reshape works correctly
        X = np.moveaxis(X, 1, 2)
        # and reshape into (batch_size, img_height, (sequence_length * img_width), num_channels)
        reshaped_X = X.reshape(self.batch_size, *self.new_image_shape)

        # y_new = np.repeat(y, self.output_size[0], axis=0)
        return reshaped_X, y
    """

    def __getitem__(self, index):
        """
        Return a new sample in the form (X, y) where X is a batch of image samples and y the corresponding labels.

        Args:
            index: the number of the current sample from 0 to __len__() - 1
        """
        X = np.empty((self.batch_size, self.sequence_length, *self.output_size), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes))
        batch_img_names = []

        # start from the current batch and take the next 'batch_size' indices
        current_index = index * self.batch_size
        indices = self.indices_list[current_index:current_index + self.batch_size]

        for i, start_index in enumerate(indices):
            sample_rows = self.df[start_index:start_index + self.sequence_length]  # get the corresponding df rows
            image_sample, sample_label, sample_names = self.__get_data(sample_rows)
            X[i, ] = image_sample
            y[i, ] = sample_label
            batch_img_names.append(sample_names)

        # move the sequence length dim to the left of the width so the reshape works correctly
        X = np.moveaxis(X, 1, 2)
        # and reshape into (batch_size, img_height, (sequence_length * img_width), num_channels)
        reshaped_X = X.reshape(self.batch_size, *self.new_image_shape)

        # y_new = np.repeat(y, self.output_size[0], axis=0)

        # print("Img names in this batch: ", batch_img_names)  # TODO save as dict: {index: batch_img_names} ?
        return reshaped_X, y

    def __get_data(self, sample):
        image_sample = np.empty((self.sequence_length, *self.output_size), dtype=np.float32)
        img_names = []

        # Load and preprocess the images for the current sample
        i = 0
        for idx, row in sample.iterrows():
            img_path = row[self.X_col]
            image_path = os.path.join(self.images_base_path, img_path)
            image_sample[i, ] = self.__scale_and_convert_image(image_path)  # load image and resize and scale it
            img_names.append(img_path)
            i += 1

        label = sample[self.y_col].iloc[0]  # take the label of the first element in the sample
        sample_label = DifficultyLevels.get_one_hot_encoding(label)  # convert string label to one-hot-vector
        # print("Before: ", image_sample[0][0][0])

        # efficientNet doesn't need preprocess_input, only input values between [0, 255]
        # TODO
        image_sample = tf.keras.applications.resnet50.preprocess_input(image_sample)
        # image_sample = tf.keras.applications.vgg16.preprocess_input(image_sample)
        # image_sample = tf.keras.applications.inception_v3.preprocess_input(image_sample)
        # image_sample = tf.keras.applications.xception.preprocess_input(image_sample)

        # print("After: ", image_sample[0][0][0])
        return image_sample, sample_label, img_names

    def __scale_and_convert_image(self, image_path):
        try:
            color_mode = "grayscale" if self.use_grayscale else "rgb"

            image = tf.keras.preprocessing.image.load_img(image_path, color_mode=color_mode)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)
            resized_img = crop_center_square(image_arr)

            # TODO different image adjustments?
            # resized_img = tf.image.adjust_hue(resized_img, -0.8)  # must be in [-1, 1]
            # resized_img = tf.image.adjust_contrast(resized_img, 1.2)
            # resized_img = tf.image.adjust_brightness(resized_img, -0.5)
            # resized_img = tf.image.adjust_saturation(resized_img, -0.5)  # TODO aus irgendeinem grund nahezu 70% manchmal (mit batchsize 12 und sample size 10); aber nur 31 auf val_data

            # normalize pixel values to [0, 1] so the CNN can work with smaller values
            # scaled_img = resized_img / 255.0
            return resized_img

        except Exception as e:
            sys.stderr.write(f"\nError in processing image '{image_path}': {e}")
            return None

    def get_image_shape(self):
        return self.new_image_shape

    def get_example_batch(self, idx=0):
        # we need to make a copy first so we don't actually change the list by taking an example
        indices_copy = self.indices_list.copy()
        first_sample, labels = self.__getitem__(idx)
        self.indices_list = indices_copy
        return first_sample, labels
