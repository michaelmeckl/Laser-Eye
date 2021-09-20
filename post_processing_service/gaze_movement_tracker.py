#!/usr/bin/python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math


SEQUENZ_LENGTH = 200

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
                columns=['participant', 'difficulty', 'left_pupil_position', 'right_pupil_position', 'left_eye_position', 'right_eye_position', 'left_eye_size_x','right_eye_size_x','left_eye_size_y','right_eye_size_y', 'time_stamp'])

    def save_eye_data_to_data_frame(self, left_pupil_position, right_pupil_position, left_eye_position, right_eye_position, left_eye_size_x,  left_eye_size_y, right_eye_size_x, right_eye_size_y, difficulty, participant, timestamp):
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

    def save_data(self, participant, difficulty):
        self.movement_df.to_csv(f"../machine_learning_predictor/SVM/data/eye_movement_data/eye_movement_{participant}_{difficulty}.csv", index=False)
        self.movement_df = pd.DataFrame(
            columns=['participant', 'difficulty', 'left_pupil_position', 'right_pupil_position', 'left_eye_position',
                     'right_eye_position', 'left_eye_size_x', 'right_eye_size_x', 'left_eye_size_y', 'right_eye_size_y',
                     'time_stamp'])