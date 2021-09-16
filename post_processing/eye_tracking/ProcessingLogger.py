import os
from datetime import datetime
from enum import Enum
from typing import Any
import cv2
import numpy as np
import pandas as pd
import pathlib
from post_processing.post_processing_constants import download_folder, post_processing_log_folder

ProcessingData = Enum("ProcessingData", "HEAD_POS_ROLL_PITCH_YAW LEFT_EYE_CENTER RIGHT_EYE_CENTER LEFT_EYE_WIDTH "
                                        "RIGHT_EYE_WIDTH LEFT_EYE_HEIGHT RIGHT_EYE_HEIGHT LEFT_PUPIL_POS "
                                        "RIGHT_PUPIL_POS LEFT_PUPIL_DIAMETER "
                                        "RIGHT_PUPIL_DIAMETER")  # FACE_LANDMARKS


# noinspection PyAttributeOutsideInit
class ProcessingLogger:
    def __init__(self):
        self.__log_folder = post_processing_log_folder

        self.__log_tag = "processing_logger"
        self.__processed_data = []
        self.__image_data = []

    def set_participant(self, participant):
        self.current_participant = participant

    def set_difficulty(self, difficulty):
        self.current_difficulty_level = difficulty
        self.__start_logging()

    def __start_logging(self):
        participant_folder = pathlib.Path(__file__).parent.parent / download_folder / self.current_participant
        self.__folder_path = participant_folder / self.__log_folder / self.current_difficulty_level
        self.__log_file_path = self.__folder_path / f"processing_log_{self.current_difficulty_level}.csv"

        # create log folder if it doesn't exist yet
        if not self.__folder_path.is_dir():
            os.makedirs(self.__folder_path)  # use 'makedirs()' to automatically create any missing parent dirs as well

    def log_frame_data(self, frame_id: float, data: dict[ProcessingData, Any]):
        # ** unpacks the dictionary as key-value pairs
        self.__processed_data.append({'date': datetime.now(), 'frame_id': frame_id, **data})

    def log_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        self.__image_data.append((dirname, filename, image, timestamp))

    def save_tracking_data(self):
        tracking_df = pd.DataFrame(self.__processed_data)
        tracking_df.to_csv(self.__log_file_path, sep=";", index=False)
        self.__processed_data.clear()  # reset tracking data

        for entry in self.__image_data:
            self.__save_image(*entry)
        self.__image_data.clear()

    def __save_image(self, dirname: str, filename: str, image: np.ndarray, timestamp: float):
        path = self.__folder_path / dirname
        if not path.is_dir():
            path.mkdir()

        # check if empty first as it crashes if given an empty array (e.g. if face / eyes not fully visible)
        if image.size:
            cv2.imwrite(f'{path / filename}__{timestamp}.png', image)
        else:
            print(f"[WARNING] Image.size is None for image: {path / filename}__{timestamp}.png")

    def log_blink_info(self, blink_info_dict):
        blink_log_path = os.path.join(self.__folder_path, f"{self.current_difficulty_level}_blink_log.csv")
        df = pd.DataFrame(blink_info_dict, index=[0])
        df.to_csv(blink_log_path, sep=",", index=False)
