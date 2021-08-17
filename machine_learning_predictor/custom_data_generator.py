import os
import random
import sys
import cv2
import numpy as np
import tensorflow as tf
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import NEW_IMAGE_SIZE


class CustomImageDataGenerator(tf.keras.utils.Sequence):
    """
    Structure based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, data_frame, x_col_name, y_col_name, batch_size, num_classes=3,
                 images_base_path=".", use_grayscale=False, shuffle=False):
        self.df = data_frame.copy()
        self.X_col = x_col_name
        self.y_col = y_col_name
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.images_base_path = images_base_path
        self.use_grayscale = use_grayscale
        self.should_shuffle = shuffle

        self.n = len(self.df)
        self.indexes = self.df.index.to_list()

        num_channels = 1 if self.use_grayscale else 3
        self.output_size = (*NEW_IMAGE_SIZE, num_channels)

        # create a random order for the samples
        self.index_order = self.generate_random_index_list()

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
        for i in range(0, self.n, self.batch_size):
            sample_indices.append(i)

        random.shuffle(sample_indices)
        return sample_indices

    """
    def generate_random_index_list(self):
        original_indexes = self.indexes[:]
        used_indices = []
        random_indexes = []
        num_images = self.n - self.batch_size
        num_samples = self.__len__()

        for i in range(num_samples-1):
            # get difference of both lists, see 
            s = set(used_indices)
            index_list = [x for x in original_indexes if x not in s]

            start_index = random.choice(range(index_list))
            random_indexes.append(start_index)
            used_indices.append(original_indexes[start_index:start_index + self.batch_size)
            num_images -= self.batch_size

        # TODO add the one batch that is left
        return random_indexes
    """

    def __len__(self):
        return self.n // self.batch_size

    # TODO test the dataframe subset version too!
    def __getitem__(self, index):
        """
        Return a new sample in the form (X, y) where X is an image and y the corresponding label.

        Args:
            index: the number of the current sample from 0 to __len__() - 1
        """
        actual_index = self.index_order[index]
        # Take all elements starting from the current index until the start of the next index
        sample_rows = self.df[actual_index:actual_index + self.batch_size]

        X, y = self.__get_data(sample_rows)
        return X, y

    def __get_data(self, sample):
        # Init empty arrays for the image and label data
        X = np.empty((self.batch_size, *self.output_size))
        y = np.empty((self.batch_size, self.n_classes))

        # Load and preprocess the images and labels for the current sample
        i = 0
        for idx, row in sample.iterrows():
            img_path = row[self.X_col]
            image_path = os.path.join(self.images_base_path, img_path)
            X[i, ] = self.__scale_and_convert_image(image_path)  # load image and resize and scale it

            label = row[self.y_col]
            y[i, ] = DifficultyLevels.get_one_hot_encoding(label)  # convert string label to one-hot-vector
            i += 1

        return X, y

    def __scale_and_convert_image(self, image_path):
        try:
            """
            # opencv version:

            # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # use this instead: (for the reason see
            # https://stackoverflow.com/questions/37203970/opencv-grayscale-mode-vs-gray-color-conversion#comment103382641_37208336)
            img = cv2.imread(image_path)
            if self.use_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            resized_img = cv2.resize(img, NEW_IMAGE_SIZE)
            scaled_img = resized_img / 255.0

            if self.use_grayscale:
                # add the third dimension again that was lost during opencv's grayscale conversion
                scaled_img = scaled_img[:, :, np.newaxis]

            return scaled_img
            """
            # tensorflow image version:
            color_mode = "grayscale" if self.use_grayscale else "rgb"

            image = tf.keras.preprocessing.image.load_img(image_path, color_mode=color_mode)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)

            # crop or pad image depending on it's size
            resized_img = tf.image.resize_with_crop_or_pad(image_arr,
                                                           target_height=NEW_IMAGE_SIZE[1],
                                                           target_width=NEW_IMAGE_SIZE[0])
            # resized_img = tf.image.resize(image_arr, [NEW_IMAGE_SIZE[1], NEW_IMAGE_SIZE[0]])

            # normalize pixel values to [0, 1] so the ml model can work with smaller values
            scaled_img = resized_img.numpy() / 255.0
            return scaled_img

        except Exception as e:
            sys.stderr.write(f"\nError in processing image '{image_path}': {e}")
            return None

    def on_epoch_end(self):
        # TODO remove used indexes from df and reset index?
        if self.should_shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_first(self):
        return self.__getitem__(0)

    def get_last(self):
        return self.__getitem__(self.__len__() - 1)

    def get_image_shape(self):
        return self.output_size
