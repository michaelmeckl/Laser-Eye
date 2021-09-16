#!/usr/bin/python3
# -*- coding:utf-8 -*-

from post_processing.convlstm_eye_region_classifier import start_training_and_testing_convlstm
from post_processing.download_data import download_data_from_server
from post_processing.efficient_net_classifier import start_training_and_testing
from post_processing.extract_downloaded_data import extract_data
from post_processing.assign_load_classes import assign_labels
from post_processing.extract_eye_features import start_extracting_eye_features


def main():
    """
    All of the functions below are the main functions in their corresponding files and can be started from there as
    well. This can be used to execute the whole process from the start in the correct order. Pass empty lists to use
    all participants or specify the names of the participant folders that should be used.
    """
    download_data_from_server(folder_names=[])
    extract_data(participant_list=[])
    assign_labels(participant_list=[])
    start_extracting_eye_features(participant_list=[], debug=False, enable_annotation=False)

    start_training_and_testing(use_eye_regions=True)  # Option 1
    # start_training_and_testing(use_eye_regions=True)  # Option 2
    # start_training_and_testing_convlstm()  # Option 3


if __name__ == "__main__":
    main()
