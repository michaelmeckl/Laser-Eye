import os
from post_processing.post_processing_constants import download_folder


RANDOM_SEED = 42
NUMBER_OF_CLASSES = 3

# NEW_IMAGE_SIZE = (112, 112)

# for VGG16, Resnet50 and EfficientNetB0:
NEW_IMAGE_SIZE = (224, 224)  # format: (height, width)
# for Inception_v3 and Xception:
# NEW_IMAGE_SIZE = (299, 299)

results_folder = "ml_results"
data_folder_path = os.path.join(os.path.dirname(__file__), "..", "post_processing", download_folder)

