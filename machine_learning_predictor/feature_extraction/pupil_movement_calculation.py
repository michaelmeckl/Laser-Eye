import pandas as pd
import numpy as np
import os
import math


class PupilMovementCalculation:

    def calculate_pupil_movement(self, is_evaluation_data: bool):
        if is_evaluation_data:
            input_dir_path = os.path.join(os.path.dirname(__file__), "data", "evaluation_eye_movement_data")
            output_dir_path = os.path.join(os.path.dirname(__file__), "data", "evaluation_pupil_movement_data")
        else:
            input_dir_path = os.path.join(os.path.dirname(__file__), "data", "eye_movement_data")
            output_dir_path = os.path.join(os.path.dirname(__file__), "data", "pupil_movement_data")

        for filename in os.listdir(input_dir_path):
            print("\n####################\nStarting pupil movement calculation\n####################\n")
            print(f"Reading in : {filename}")
            pupil_movement_df = pd.DataFrame(columns=['participant', 'difficulty', 'left_pupil_movement_x',
                                                      'left_pupil_movement_y',
                                                      'right_pupil_movement_x',
                                                      'right_pupil_movement_y',
                                                      'average_pupil_movement_x',
                                                      'average_pupil_movement_y',
                                                      'average_pupil_movement_distance', 'movement_angle',
                                                      'strong_movement', 'direction_change', 'time_difference',
                                                      'time_stamp'])

            eye_movement_df_path = os.path.join(input_dir_path, filename)
            eye_movement_df = pd.read_csv(eye_movement_df_path, converters={"column_name": eval})

            rowCount = eye_movement_df.shape[0]
            print(f"length: {rowCount}")
            last_left_pupil_position = None
            last_right_pupil_position = None
            last_left_eye_position = None
            last_right_eye_position = None
            last_time_stamp = None
            first_row = True
            for index, row in eye_movement_df.iterrows():

                # show progress
                if index % int(rowCount / 10) == 0:
                    print(f"Calculating pupil movement done: {index / int(rowCount / 100):.2f} %")

                left_pupil_position = row["left_pupil_position"][1:-1].split(", ")

                right_pupil_position = row["right_pupil_position"][1:-1].split(", ")

                left_eye_position = row["left_eye_position"][1:-1].split(", ")

                right_eye_position = row["right_eye_position"][1:-1].split(", ")

                left_eye_size_x = row["left_eye_size_x"]
                right_eye_size_x = row["right_eye_size_x"]
                left_eye_size_y = row["left_eye_size_y"]
                right_eye_size_y = row["right_eye_size_y"]
                difficulty = row["difficulty"]
                participant = row["participant"]
                timestamp = row["time_stamp"]

                if not first_row:
                    #   calculate change in pupil position depending on eye position, normalized to eye size

                    x_movement_left_pupil = ((float(last_left_pupil_position[0]) - float(last_left_eye_position[0])) - (
                            float(left_pupil_position[0]) - float(left_eye_position[0]))) / left_eye_size_x
                    y_movement_left_pupil = ((float(last_left_pupil_position[1]) - float(last_left_eye_position[1])) - (
                            float(left_pupil_position[1]) - float(left_eye_position[1]))) / left_eye_size_y

                    left_pupil_movement = (x_movement_left_pupil, y_movement_left_pupil)

                    x_movement_right_pupil = ((float(last_right_pupil_position[0]) - float(
                        last_right_eye_position[0])) - (float(right_pupil_position[0]) - float(right_eye_position[0]))) / right_eye_size_x

                    y_movement_right_pupil = ((float(last_right_pupil_position[1]) - float(
                        last_right_eye_position[1])) - (float(right_pupil_position[1]) - float(right_eye_position[1]))) / right_eye_size_y

                    right_pupil_movement = (x_movement_right_pupil, y_movement_right_pupil)

                    average_pupil_movement = [(x_movement_right_pupil + x_movement_left_pupil) / 2,
                                              (y_movement_right_pupil + y_movement_left_pupil) / 2]
                    average_pupil_movement_distance = np.sqrt(
                        average_pupil_movement[0] ** 2 + average_pupil_movement[1] ** 2)

                    # Calculate time difference inbetween images (this is outdated, since we "interpolated"
                    # images to a time difference of 100ms)

                    time_difference = None
                    time_difference = int(timestamp) - int(last_time_stamp)

                    # Calculate angle of the eye_movement
                    standard_vector = (1, 0)
                    angle = self.__calculate_angle(standard_vector, average_pupil_movement)

                    # Check if it was a strong movement and if the direction changed by more than 90° and less than 270°
                    direction_change = 0
                    if 270 > angle > 90:
                        direction_change = 1
                    strong_movement = 0
                    if average_pupil_movement_distance > 0.02:
                        strong_movement = 1

                    # Write data to dataframe
                    pupil_movement_df = pupil_movement_df.append({'participant': participant, 'difficulty': difficulty,
                                                                  'left_pupil_movement_x': left_pupil_movement[0],
                                                                  'left_pupil_movement_y': left_pupil_movement[1],
                                                                  'right_pupil_movement_x': right_pupil_movement[0],
                                                                  'right_pupil_movement_y': right_pupil_movement[1],
                                                                  'average_pupil_movement_x': average_pupil_movement[0],
                                                                  'average_pupil_movement_y': average_pupil_movement[1],
                                                                  'average_pupil_movement_distance': average_pupil_movement_distance,
                                                                  'movement_angle': angle,
                                                                  'strong_movement': strong_movement,
                                                                  'direction_change': direction_change,
                                                                  'time_difference': time_difference,
                                                                  'time_stamp': timestamp},
                                                                 ignore_index=True)
                # For the first row we add empty values into the dataframe
                else:
                    pupil_movement_df = pupil_movement_df.append({'participant': participant, 'difficulty': difficulty,
                                                                  'left_pupil_movement_x': 0,
                                                                  'left_pupil_movement_y': 0,
                                                                  'right_pupil_movement_x': 0,
                                                                  'right_pupil_movement_y': 0,
                                                                  'average_pupil_movement_x': 0,
                                                                  'average_pupil_movement_y': 0,
                                                                  'average_pupil_movement_distance': 0,
                                                                  'movement_angle': 0,
                                                                  'strong_movement': 0,
                                                                  'direction_change': 0,
                                                                  'time_difference': 0,
                                                                  'time_stamp': timestamp},
                                                                 ignore_index=True)
                first_row = False
                last_left_pupil_position = left_pupil_position
                last_right_pupil_position = right_pupil_position
                last_left_eye_position = left_eye_position
                last_right_eye_position = right_eye_position
                last_time_stamp = timestamp

            if not os.path.exists(output_dir_path):
                os.mkdir(output_dir_path)

            pupil_movement_path = os.path.join(output_dir_path, f"pupil_movement_{participant}_{difficulty}.csv")
            pupil_movement_df.to_csv(pupil_movement_path)

        print("\n####################\nFinished pupil movement calculation!\n####################\n")

    def __calculate_angle(self, p1, p2):
        # ang1 = np.arctan2(*p1[::-1])
        # ang2 = np.arctan2(*p2[::-1])
        # return np.rad2deg((ang1 - ang2) % (2 * np.pi))

        vector_1 = p1

        vector_2 = p2

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)

        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)

        dot_product = np.dot(unit_vector_1, unit_vector_2)

        angle = np.arccos(dot_product)
        if not math.isnan(angle):
            return angle
        else:
            return 0

    def calculate_frequencies(self, data_array):
        try:
            # highest_quartile_list = sorted((data_array))[int(len(data_array)*0.75):]
            # lowest_quartile_list = sorted((data_array))[0:int(len(data_array) * 0.25)]
            # data_array = np.setdiff1d(data_array, highest_quartile_list)
            # data_array = np.setdiff1d(data_array, lowest_quartile_list)

            # fft computing and normalization
            fourier = np.abs(np.fft.fft(data_array))
            return fourier[0:int(len(fourier))]
        except Exception as e:
            print(f"Exception when trying to perform fast fourier transformation: {e}\n")
            return data_array
