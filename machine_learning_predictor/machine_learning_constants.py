import os
from post_processing.post_processing_constants import download_folder


RANDOM_SEED = 42
NUMBER_OF_CLASSES = 3
NEW_IMAGE_SIZE = (128, 128)  # (256, 256)  # TODO or try to keep the ratio of the images -> should be taller than wide

TRAIN_SUBSET_SIZE = 450

results_folder = "ml_results"
data_folder_path = os.path.join(os.path.dirname(__file__), "..", "post_processing", download_folder)

