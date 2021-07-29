#!/usr/bin/python3
# -*- coding:utf-8 -*-

from post_processing.download_data import download_data_from_server
from post_processing.process_downloaded_data import extract_data
from post_processing.assign_load_classes import assign_load
from post_processing.extract_eye_features import start_extracting_features


def main():
    """
    All of the functions below are the main functions in their corresponding files and can be started from there as
    well. This can be used to execute the whole process from the start in the correct order.
    """
    download_data_from_server(folder_names=[])
    extract_data(participant_list=[])
    assign_load()
    start_extracting_features(debug=False)


if __name__ == "__main__":
    main()
    """
    archive = py7zr.SevenZipFile(pathlib.Path(__file__).parent.parent / "game_log.7z", mode="r")
    archive.extractall(path=pathlib.Path(__file__).parent.parent)
    archive.close()
    """
