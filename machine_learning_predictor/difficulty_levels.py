import sys
from enum import Enum
from typing import Union, Iterable
import numpy as np


class DifficultyLevels(Enum):
    HARD = "hard"
    MEDIUM = "medium"
    EASY = "easy"

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))

    @staticmethod
    def get_one_hot_encoding(category):
        if category not in DifficultyLevels.values():
            sys.stderr.write(f"\nNo one hot encoding possible for category {category}. Must be one of '"
                             f"{DifficultyLevels.values()}'!")
            return None

        # return one-hot-encoded vector for this category
        if category == DifficultyLevels.HARD.value:
            label_vector = [0, 0, 1]
        elif category == DifficultyLevels.MEDIUM.value:
            label_vector = [0, 1, 0]
        elif category == DifficultyLevels.EASY.value:
            label_vector = [1, 0, 0]
        else:
            sys.stderr.write(f"\nCategory {category} doesn't match one of the difficulty levels!")
            return None

        return label_vector

    @staticmethod
    def get_label_for_encoding(encoded_vector: Union[np.ndarray, Iterable]):
        # np.array_equal() as the encoded vector can either be a normal list or a ndarray
        if np.array_equal(encoded_vector, [0, 0, 1]):
            label = DifficultyLevels.HARD.value
        elif np.array_equal(encoded_vector, [0, 1, 0]):
            label = DifficultyLevels.MEDIUM.value
        elif np.array_equal(encoded_vector, [1, 0, 0]):
            label = DifficultyLevels.EASY.value
        else:
            sys.stderr.write(f"\nEncoded_vector {encoded_vector} can't be matched to one of the labels!")
            return None

        return label
