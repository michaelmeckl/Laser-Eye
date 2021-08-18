import os
import random
import sys
import numpy as np
import tensorflow as tf
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import NEW_IMAGE_SIZE


class CustomImageDataGenerator(tf.keras.utils.Sequence):
    """
    Structure based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, data_frame, x_col_name, y_col_name, sample_size, batch_size, num_classes=3,
                 images_base_path=".", use_grayscale=False):
        self.df = data_frame.copy()
        # self.original_df = data_frame.copy()
        self.X_col = x_col_name
        self.y_col = y_col_name
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.images_base_path = images_base_path
        self.use_grayscale = use_grayscale

        self.n = len(self.df)
        self.indices = self.df.index.to_list()

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
        for i in range(0, self.n, self.sample_size):
            sample_indices.append(i)

        random.shuffle(sample_indices)
        return sample_indices

    """
    def create_random_index_list(self):
        original_indexes = self.indices[:]
        used_indices = []
        random_indexes = []
        num_samples = self.__len__()

        for i in range(num_samples):
            # index_list = [x for x in original_indexes if x not in set(used_indices)]

            start_index = random.choice(original_indexes)
            random_indexes.append(start_index)
            del original_indexes[start_index:start_index + self.batch_size]
            # used_indices.extend(original_indexes[start_index:start_index + self.batch_size])

        # add the one batch that is left
        # index_list = [x for x in original_indexes if x not in set(used_indices)]
        # last_sample_index = index_list[0]
        # random_indexes.append(last_sample_index)

        random.shuffle(random_indexes)
        return random_indexes
    """

    def __len__(self):
        return self.n // self.sample_size

    def on_epoch_end(self):
        random.shuffle(self.index_order)
        # self.df = self.original_df.copy()
        # # self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        """
        Return a new sample in the form (X, y) where X is an image and y the corresponding label.

        Args:
            index: the number of the current sample from 0 to __len__() - 1
        """
        actual_index = self.index_order[index]
        # actual_index = random.choice(self.df.index.to_list())

        # Take all elements starting from the current index until the start of the next index
        sample_rows = self.df[actual_index:actual_index + self.sample_size]

        X, y = self.__get_data(sample_rows)
        # TODO only works if workers are set to 1 in classifier!
        # self.df.drop(self.df.index[actual_index:actual_index + self.batch_size], inplace=True)
        # if len(self.df) == 0:
        #     self.df = self.original_df.copy()
        # self.df = self.df.reset_index(drop=True)

        return X, y

    def __get_data(self, sample):
        # Setup arrays for the image and label data
        X = np.empty((self.sample_size, *self.output_size))
        y = np.empty((self.sample_size, self.n_classes))

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
            color_mode = "grayscale" if self.use_grayscale else "rgb"

            image = tf.keras.preprocessing.image.load_img(image_path, color_mode=color_mode)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)

            # crop or pad image depending on it's size
            resized_img = tf.image.resize_with_crop_or_pad(image_arr,
                                                           target_height=NEW_IMAGE_SIZE[1],
                                                           target_width=NEW_IMAGE_SIZE[0])

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
