#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import pandas as pd


# SEQUENCE_LENGTH = 200


class GazeMovementTracker:
    def __init__(self):
        self.left_pupil_movements = []
        self.right_pupil_movements = []
        self.last_left_pupil_position = None
        self.last_right_pupil_position = None
        self.last_left_eye_position = None
        self.last_right_eye_position = None
        self.last_time_stamp = None
        self.pupil_movement_array = []
        self.difficulties = []
        self.movement_df = pd.DataFrame(
            columns=['participant', 'difficulty', 'left_pupil_position', 'right_pupil_position', 'left_eye_position',
                     'right_eye_position', 'left_eye_size_x', 'right_eye_size_x', 'left_eye_size_y', 'right_eye_size_y',
                     'time_stamp'])

    def save_eye_data_to_data_frame(self, left_pupil_position, right_pupil_position, left_eye_position,
                                    right_eye_position, left_eye_size_x, left_eye_size_y, right_eye_size_x,
                                    right_eye_size_y, difficulty, participant, timestamp):

        self.movement_df = self.movement_df.append({'participant': participant, 'difficulty': difficulty,
                                                    'left_pupil_position': left_pupil_position,
                                                    'right_pupil_position': right_pupil_position,
                                                    'left_eye_position': left_eye_position,
                                                    'right_eye_position': right_eye_position,
                                                    'left_eye_size_x': left_eye_size_x,
                                                    'right_eye_size_x': right_eye_size_x,
                                                    'left_eye_size_y': left_eye_size_y,
                                                    'right_eye_size_y': right_eye_size_y,
                                                    'time_stamp': timestamp},
                                                   ignore_index=True)

    def save_data(self, participant, difficulty, evaluation_study_data: bool):
        # use os.path.dirname(__file__) to get the correct path to this file independent of the calling location
        ml_folder = os.path.join(os.path.dirname(__file__), "..", "machine_learning_predictor")
        if evaluation_study_data:
            output_eye_data_folder = os.path.join(ml_folder, "feature_extraction", "data",
                                                  "evaluation_eye_movement_data")
        else:
            output_eye_data_folder = os.path.join(ml_folder, "feature_extraction", "data", "eye_movement_data")

        if not os.path.exists(output_eye_data_folder):
            os.makedirs(output_eye_data_folder)

        eye_movement_path = os.path.join(output_eye_data_folder, f"eye_movement_{participant}_{difficulty}.csv")
        self.movement_df.to_csv(eye_movement_path, index=False)
        self.movement_df = pd.DataFrame(
            columns=['participant', 'difficulty', 'left_pupil_position', 'right_pupil_position', 'left_eye_position',
                     'right_eye_position', 'left_eye_size_x', 'right_eye_size_x', 'left_eye_size_y', 'right_eye_size_y',
                     'time_stamp'])
