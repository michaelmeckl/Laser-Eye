import os

download_folder = "tracking_data_download"
image_folder = "extracted_images"
logs_folder = "extracted_logs"
post_processing_log_folder = "post_processing_data"

blur_threshold = 25  # determined by testing with different participants

evaluation_study_folder = "evaluation_study"
evaluation_download_folder = os.path.join(os.path.dirname(__file__), evaluation_study_folder, "evaluation_study_data")
