#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import csv
import shutil
import sys
import cv2
import pandas as pd
from collections import defaultdict
from bisect import bisect_left
from post_processing.post_processing_constants import download_folder, blur_threshold

data_folder = download_folder
start_events = ["started", "Game_Start"]
end_events = ["ended", "Game_End"]


def merge_csv_logs(game_log_folder):
    all_game_logs = pd.DataFrame()

    for log_file in os.listdir(game_log_folder):
        if "GameLog" in log_file or "Tetris" in log_file or ("Clicking" in log_file and "Missed" not in log_file):
            # get header names
            with open(os.path.join(game_log_folder, log_file), 'r') as f:
                header = next(csv.reader(f))
            # drop one specific column in pandas, see
            # https://stackoverflow.com/questions/48899051/how-to-drop-a-specific-column-of-csv-file-while
            # -reading-it-using-pandas/48899141
            csv_content = pd.read_csv(os.path.join(game_log_folder, log_file), usecols=list(set(header) - {'args'}))

            # remove whitespaces in column headers and csv cells, see
            # https://stackoverflow.com/questions/49551336/pandas-trim-leading-trailing-white-space-in-a-dataframe
            csv_content.columns = csv_content.columns.str.strip()
            cols = csv_content.select_dtypes(['object']).columns
            csv_content[cols] = csv_content[cols].apply(lambda x: x.str.strip())

            all_game_logs = pd.concat([all_game_logs, csv_content])

    # fix typo in column header
    all_game_logs.rename(columns={"eventy_type": "event_type"}, inplace=True)
    return all_game_logs


def assert_task_duration(start_time, end_time, error_variance=10):
    """
    Make sure the time difference between the start and end event type for each difficulty and game is 100 seconds (+
    some small measurement error).
    """
    duration = (end_time - start_time) / 1000  # get duration in seconds; start and end time are in ms
    # assert that duration is around 100 seconds
    assert (100 - error_variance) <= duration <= (100 + error_variance), f"Time difference between start and end " \
                                                                         f"should be ca. 100 seconds but was {duration}"


def perform_checks(df: pd.DataFrame):
    """
    Performs some sanity checks on the combined dataframe per participant to make sure everything worked as expected.
    """

    print("len rows:", len(df.index))
    assert len(df.index) == 30  # 5 games * 3 difficulties each * 2 events (start & end)

    num_start_events = len(df[df['event_type'].isin(start_events)])
    num_end_events = len(df[df['event_type'].isin(end_events)])
    assert num_start_events == num_end_events, "The number of start and end events don't match!!"

    assert df["gameid"].nunique() == 5, "Not all games exist for this participant!"

    for game_name in df["gameid"].unique():
        # each name should appear 6 times: 3 difficulties * 2 (start & end event type for each)
        assert len(df[df['gameid'] == game_name]) == 6, f"The number of occurences for game '{game_name}' isn't 6!"

    row_iterator = df.iterrows()
    _, last = next(row_iterator)  # take first item from row_iterator
    for i, row in row_iterator:
        curr_event_type = row['event_type']
        previous_event_type = last['event_type']
        if previous_event_type in start_events:
            assert curr_event_type in end_events, "Current event type is not end but previous was start!"

            # if the previous event type was start and the current one is end, assert that the time between both is
            # the duration we have set for each difficulty (100 seconds)
            curr_timestamp = row['timestamp']
            previous_timestamp = last['timestamp']
            assert_task_duration(previous_timestamp, curr_timestamp)

        elif previous_event_type in end_events:
            assert curr_event_type in start_events, "Current event type is not start but previous was end!"

        last = row


def take_closest(search_list, search_number):
    """
    Assumes the list is sorted. Returns closest value to search_number.
    If two numbers are equally close, return the smallest number.

    Taken from https://stackoverflow.com/a/12141511
    """
    pos = bisect_left(search_list, search_number)
    if pos == 0:
        return search_list[0]
    if pos == len(search_list):
        return search_list[-1]
    before = search_list[pos - 1]
    after = search_list[pos]
    if after - search_number < search_number - before:
        return after
    else:
        return before


def find_closest_value(dictionary, timestamp):
    # get the value if it exists, if not find the closest key in the dictionary for this timestamp and return its value
    return dictionary.get(timestamp) or dictionary.get(take_closest(list(dictionary.keys()), timestamp))


def find_image_event_indexes(df, all_images, sorted_img_dict):
    """
    Find the corresponding images for the difficulty start and end times.
    """
    result_dict = defaultdict(list)
    start_pos = None
    end_pos = None
    for i, row in df.iterrows():
        event_type = row["event_type"]
        difficulty = row["difficulty"]
        timestamp = row["timestamp"]

        if event_type in start_events:
            # get doesn't work as there won't be any values that match exactly (no image is captured at the
            # exact same milliseconds than the game start or game end events were logged)
            # Because of this we need to find the closest value from the image timestamps!
            start_pos = find_closest_value(sorted_img_dict, timestamp)
        elif event_type in end_events:
            end_pos = find_closest_value(sorted_img_dict, timestamp)
            if start_pos and end_pos:
                image_slice = all_images[start_pos:end_pos]
                result_dict[difficulty].extend(image_slice)
            elif end_pos < start_pos:
                sys.stderr.write("Something went horribly wrong... End pos should always be larger than start pos!")

    return result_dict


def check_image_blur(image_path) -> float:
    # Function taken from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    image = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_scale, cv2.CV_64F).var()


def split_image_folder(result_dict, participant_folder):
    """
    Param `result_dict` contains a list for each of the 3 load levels which contains all images from this participant
    that were recorded during a game with the corresponding difficulty:
    ```
    {
        'hard': [capture__1223445.34.png, capture__122673445.89.png, ...],
        'medium': [...],
        'easy': [...],
    }
    ```
    """

    print("Number images 'hard':", len(result_dict['hard']))
    print("Number images 'medium':", len(result_dict['medium']))
    print("Number images 'easy':", len(result_dict['easy']))
    # pd.DataFrame.from_dict(result_dict).to_csv(os.path.join(data_folder, participant_folder))

    # TODO result_dict mit pd.to_csv in eine csv datei speichern und da für jeden participant + difficulty eine
    #  row anlegen und ganzen pfad zum bild speichern ?
    labeled_images_folder = os.path.join(data_folder, participant_folder, "labeled_images")
    if not os.path.exists(labeled_images_folder):
        os.mkdir(labeled_images_folder)

    for difficulty, image_list in result_dict.items():
        difficulty_folder = os.path.join(labeled_images_folder, difficulty)
        if not os.path.exists(difficulty_folder):
            os.mkdir(difficulty_folder)

            for image_name in image_list:
                original_image_path = os.path.join(data_folder, participant_folder, "extracted_images", image_name)
                """
                # skip images that are too blurred
                focus_measure = check_image_blur(original_image_path)
                if focus_measure <= blur_threshold:
                    continue
                """
                new_image_path = os.path.join(difficulty_folder, image_name)
                shutil.copy2(original_image_path, new_image_path)
                # shutil.move(original_image_path, new_image_path)  # way faster than copy but changes the dir structure
        else:
            print(f"Folder {difficulty_folder} already exists. Skipping...")

    # shutil.rmtree(os.path.join(data_folder, participant_folder, "extracted_images"))


def assign_load(participant_list=list[str]):
    for participant in os.listdir(data_folder):
        # if specific participants are given, skip the others
        if len(participant_list) > 0 and participant not in participant_list:
            print(f"\nSkipping folder {participant} as it is not in the specified participant list.\n")
            continue

        print(f"\n####################\nAssigning labels for participant {participant}\n####################\n")
        images_folder = os.path.join(data_folder, participant, "extracted_images")
        game_log_folder = os.path.join(data_folder, participant, "extracted_logs", "CSV")

        # iterate over all logs for all games and concatenate them into one large pandas dataframe
        all_game_logs = merge_csv_logs(game_log_folder)

        # remove the tutorial rows
        all_logs = all_game_logs[all_game_logs.difficulty != 'tutorial']
        # and sort the df based on the timestamp column in ascending order
        all_logs_sorted = all_logs.sort_values('timestamp')
        # print(all_logs_sorted['timestamp'].equals(all_game_logs['timestamp']))

        # remove unnecessary columns
        df_start_end = all_logs_sorted[all_logs_sorted['event_type'].isin(start_events + end_events)]
        df_start_end_small = df_start_end.drop(['state_id', 'seed', 'score', 'args'], axis=1)
        # print(df_start_end_small.head(30))

        # some sanity checks and assertions to make sure the logs are correct
        # perform_checks(df_start_end_small)  # TODO enable again later

        # extract all image timestamps from the image names and save them with the image index position
        all_images = os.listdir(images_folder)
        img_dict = {}
        for idx, image_file in enumerate(all_images):
            # all images are in the format: "capture__timestamp_part1.timestamp_part2.png"
            img_timestamp = image_file.removesuffix(".png").split("__")[1]
            timestamp = img_timestamp.split(".")[0]
            img_dict[timestamp] = idx

        # convert the keys from string to int
        img_dict = {int(key): value for key, value in img_dict.items()}
        # sort the image timestamp as well (even though the order should be already correct right now but whatever)
        # see https://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key#comment89671526_9001529
        sorted_img_dict = dict(sorted(img_dict.items()))

        result_dict = find_image_event_indexes(df_start_end_small, all_images, sorted_img_dict)
        split_image_folder(result_dict, participant)


if __name__ == "__main__":
    assign_load(participant_list=["participant_4"])
