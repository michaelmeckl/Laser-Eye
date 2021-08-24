import os
import random
import sys
import numpy as np
import tensorflow as tf
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import NEW_IMAGE_SIZE, NUMBER_OF_CLASSES


class CustomImageDataGenerator(tf.keras.utils.Sequence):
    """
    Structure based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, data_frame, x_col_name, y_col_name, sequence_length, batch_size, num_classes=NUMBER_OF_CLASSES,
                 images_base_path=".", use_grayscale=False):

        self.df = data_frame.copy()
        self.X_col = x_col_name
        self.y_col = y_col_name
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.images_base_path = images_base_path
        self.use_grayscale = use_grayscale

        self.n = len(self.df)
        # self.indices = self.df.index.to_list()
        self.get_item_index = None

        num_channels = 1 if self.use_grayscale else 3
        self.output_size = (*NEW_IMAGE_SIZE, num_channels)

        # create a random order for the samples
        self.indices_list = self.generate_random_index_list()

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
        # if length * self.batch_size * self.sequence_length < self.n:
        #    length += 1
        return length

    def on_epoch_end(self):
        self.indices_list = self.generate_random_index_list()  # each epoch we generate a new indices order
        print(f"\nEpoch finished! Generating new indices list: {self.indices_list}\n", flush=True)

    def __getitem__(self, index):
        """
        Return a new sample in the form (X, y) where X is a batch of image samples and y the corresponding labels.

        Args:
            index: the number of the current sample from 0 to __len__() - 1
        """
        self.get_item_index = index  # TODO only for debugging
        new_image_shape = ((self.sequence_length * self.output_size[0]), self.output_size[1], self.output_size[2])

        X = np.empty((self.batch_size, *new_image_shape))
        y = np.empty((self.batch_size, self.n_classes))

        if len(self.indices_list) < self.batch_size:
            print(f"\nself.indices_list is smaller than batch size in __get_item__()!", flush=True)
            # self.indices_list = self.generate_random_index_list()  # TODO

        indices = random.sample(self.indices_list, k=self.batch_size)  # get k (= batch_size) random elements from list
        for i, start_index in enumerate(indices):
            sample_rows = self.df[start_index:start_index + self.sequence_length]  # get the corresponding df rows
            image_sample, sample_label = self.__get_data(sample_rows)

            reshaped_image_sample = image_sample.reshape(new_image_shape)  # reshape into (sequence_length * img_height)
            X[i, ] = reshaped_image_sample
            y[i, ] = sample_label

        # X_new = tf.concat([X[i] for i in range(len(X))], axis=0)

        # y_new = np.repeat(y, self.output_size[0], axis=0)
        # print(y_new.shape)

        # removed the used indices from the original list so they won't be selected more than once per epoch
        self.indices_list = [idx for idx in self.indices_list if idx not in set(indices)]
        return X, y

    def __get_data(self, sample):
        image_sample = np.empty((self.sequence_length, *self.output_size))

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
            # TODO
            # image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)

            # crop or pad image depending on it's size
            resized_img = tf.image.resize_with_crop_or_pad(image_arr,
                                                           target_height=NEW_IMAGE_SIZE[0],
                                                           target_width=NEW_IMAGE_SIZE[1])

            # normalize pixel values to [0, 1] so the ml model can work with smaller values
            scaled_img = resized_img.numpy() / 255.0
            return scaled_img

        except Exception as e:
            sys.stderr.write(f"\nError in processing image '{image_path}': {e}")
            return None

    def get_first(self):
        return self.__getitem__(0)

    def get_last(self):
        return self.__getitem__(self.__len__() - 1)

    def get_image_shape(self):
        return self.output_size

    def get_example_batch(self):
        # we need to make a copy first so we don't actually change the list by taking an example
        indices_copy = self.indices_list.copy()
        first_sample, labels = self.__getitem__(0)
        self.indices_list = indices_copy
        return first_sample, labels
