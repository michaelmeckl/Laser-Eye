#!/usr/bin/python3
# -*- coding:utf-8 -*-

from machine_learning_predictor.feature_extraction.pupil_movement_calculation import PupilMovementCalculation
from post_processing.convlstm_eye_region_classifier import start_training_and_testing_convlstm
from post_processing.download_data import download_data_from_server
from post_processing.efficient_net_classifier import start_training_and_testing
from post_processing.evaluation_study.assign_difficulty_labels_evaluation import assign_evaluation_labels
from post_processing.evaluation_study.download_evaluation_data import download_evaluation_data_from_server
from post_processing.evaluation_study.evaluation_study_data_extractor import extract_evaluation_data
from post_processing.evaluation_study.extract_eye_features_evaluation import start_extracting_evaluation_eye_features
from post_processing.extract_downloaded_data import extract_data
from post_processing.assign_load_classes import assign_labels
from post_processing.extract_eye_features import start_extracting_eye_features


def start_train_pipeline():
    """
    All of the functions below are the main functions in their corresponding files and can be started from there as
    well. This can be used to execute the whole process from the start in the correct order. Pass empty lists to use
    all participants or specify the names of the participant folders that should be used.
    """
    download_data_from_server(folder_names=[])
    extract_data(participant_list=[])
    assign_labels(participant_list=[])
    start_extracting_eye_features(participant_list=[], debug=False, enable_annotation=False)

    pupil_movement_calculator = PupilMovementCalculation()
    pupil_movement_calculator.calculate_pupil_movement(is_evaluation_data=False)

    # start_training_and_testing(use_eye_regions=False)  # Option 1
    # start_training_and_testing(use_eye_regions=True)  # Option 2
    # start_training_and_testing_convlstm()  # Option 3


def start_evaluation_pipeline():
    download_evaluation_data_from_server(folder_names=[])
    extract_evaluation_data(participant_list=[])
    assign_evaluation_labels(participant_list=[])
    start_extracting_evaluation_eye_features(participant_list=[], enable_annotation=False)

    pupil_movement_calculator = PupilMovementCalculation()
    pupil_movement_calculator.calculate_pupil_movement(is_evaluation_data=True)


if __name__ == "__main__":
    # start_train_pipeline()
    start_evaluation_pipeline()
