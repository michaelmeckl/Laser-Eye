#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import csv
import shutil
import sys
import time
import cv2
import pandas as pd
from collections import defaultdict
from bisect import bisect_left
from post_processing.post_processing_constants import download_folder, image_folder, logs_folder, blur_threshold
from post_processing.process_downloaded_data import get_smallest_fps


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


def assert_task_duration(start_time, end_time, error_variance=12):
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

    number_of_games = df["gameid"].nunique()
    if number_of_games != 5:
        print(f"[WARNING]: Only {number_of_games} games found!")
        # if we not 5 games, check if we have 3 or 4 instead (as we may have removed the centaur game and/or asteroids)
        assert df["gameid"].nunique() in [3, 4], f"Only {number_of_games} found for this participant!"

    print("len rows:", len(df.index))
    assert len(df.index) == number_of_games * 3 * 2  # number of games * 3 difficulties each * 2 events (start & end)

    num_start_events = len(df[df['event_type'].isin(start_events)])
    num_end_events = len(df[df['event_type'].isin(end_events)])
    assert num_start_events == num_end_events, "The number of start and end events don't match!!"

    for game_name in df["gameid"].unique():
        # each name should appear 6 times: 3 difficulties * 2 (start & end event type for each)
        assert len(df[df['gameid'] == game_name]) == 6, f"The number of occurences for game '{game_name}' isn't 6!"

    # check that the event types (and therefore the timestamps) of game start and end are in the correct order
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


def get_timestamp_from_image(image_file_name):
    # all images are in the format: "capture__timestamp.timestamp_nanosec_precision.png"
    img_timestamp = image_file_name.removesuffix(".png").split("__")[1]
    timestamp = img_timestamp.split(".")[0]  # we only want the timestamp in ms precision to match the game logs
    return timestamp


def normalize_images_per_participant(sorted_img_file_dict, current_timestamp, start_timestamp, time_dist=100):
    time_diff_overall = current_timestamp - start_timestamp
    image_list = []

    for t in range(0, time_diff_overall, time_dist):
        new_time = start_timestamp + t
        image = find_closest_value(sorted_img_file_dict, new_time)
        image_list.append(image)

    return image_list

"""
for difficulty, image_list in original_dict.items():
    new_image_list = []
    last_timestamp = None
    last_timestamp_image = None

    for image in image_list:
        timestamp = int(get_timestamp_from_image(image))
        if last_timestamp is None:
            new_image_list.append(image)
            last_timestamp = timestamp
            last_timestamp_image = image
            continue

        if (timestamp - last_timestamp) >= time_dist:
            new_image_list.append(image)
            last_timestamp = timestamp
            last_timestamp_image = image
        else:
            print("Timestamp Diff wasn't greater or equal to time dist!")
            # new_image_list.append(last_timestamp_image) # append the last image?  # TODO no! append the closest?

    new_result_dict[difficulty].extend(new_image_list)
"""


def find_image_event_indexes(df, all_images, sorted_img_idx_dict, sorted_img_file_dict, fps_val):
    """
    Find the corresponding images for the difficulty start and end times.
    """

    # we take an image every 'time_dist' milliseconds to get an equal amount of images per participant and difficulty
    time_dist = round(1000 / fps_val)

    result_dict = defaultdict(list)
    start_pos_timestamp = None
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
            start_pos = find_closest_value(sorted_img_idx_dict, timestamp)
            start_pos_timestamp = timestamp
        elif event_type in end_events:
            end_pos = find_closest_value(sorted_img_idx_dict, timestamp)
            if start_pos and end_pos:
                # image_slice = all_images[start_pos:end_pos]
                new_image_slice = normalize_images_per_participant(sorted_img_file_dict, timestamp,
                                                                   start_pos_timestamp, time_dist)
                result_dict[difficulty].extend(new_image_slice)
            elif end_pos < start_pos:
                sys.stderr.write("Something went horribly wrong... End pos should always be larger than start pos!")

    return result_dict


def check_image_blur(image_path) -> float:
    # Function taken from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    image = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_scale, cv2.CV_64F).var()


def split_image_folder(result_dict, participant_folder, create_new_folders=False):
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
    start_time = time.time()

    # create a pandas df with 2 columns where each row consists of the image path and the corresponding load level
    label_df = pd.DataFrame(columns=["image_path", "participant", "difficulty"])
    for difficulty, image_list in result_dict.items():
        label_df = label_df.append({"image_path": image_list, "participant": participant_folder,
                                    "difficulty": difficulty}, ignore_index=True)

    # right now, the first column contains only a list with all paths, so this column needs to be "exploded",
    # see https://stackoverflow.com/questions/53218931/how-to-unnest-explode-a-column-in-a-pandas-dataframe
    df_exploded = label_df.explode("image_path")

    # add download and participant_folder to the path column to have complete paths
    df_exploded["image_path"] = df_exploded["image_path"].apply(
        lambda name: f"{os.path.join(download_folder, participant_folder, image_folder, name)}"
    )
    df_exploded.to_csv(os.path.join(data_folder, participant_folder, "labeled_images.csv"), index=False)

    end_time = time.time()
    print(f"Writing csv for {participant_folder} took {end_time-start_time} seconds.")

    # if this flag is set create a new folder "labeled_images" with the difficulty levels as subfolders
    if create_new_folders is True:
        labeled_images_folder = os.path.join(data_folder, participant_folder, "labeled_images")
        if not os.path.exists(labeled_images_folder):
            os.mkdir(labeled_images_folder)

        for difficulty, image_list in result_dict.items():
            difficulty_folder = os.path.join(labeled_images_folder, difficulty)
            if not os.path.exists(difficulty_folder):
                os.mkdir(difficulty_folder)

                for image_name in image_list:
                    original_image_path = os.path.join(data_folder, participant_folder, image_folder, image_name)
                    # TODO enable blur check to remove too blurry images?
                    """
                    # skip images that are too blurred
                    focus_measure = check_image_blur(original_image_path)
                    if focus_measure <= blur_threshold:
                        continue
                    """
                    new_image_path = os.path.join(difficulty_folder, image_name)
                    shutil.copy2(original_image_path, new_image_path)
                    # move is way faster than copy but removes the images from their old directory
                    # shutil.move(original_image_path, new_image_path)
            else:
                print(f"Folder {difficulty_folder} already exists. Skipping...")


def assign_load(participant_list=list[str]):
    smallest_fps_val = get_smallest_fps()

    smallest_difficulty_list_len = None  # length of the list with the least entries over all participants
    participant_dict = defaultdict(dict)

    for participant in os.listdir(data_folder):
        # if specific participants are given, skip the others
        if len(participant_list) > 0 and participant not in participant_list:
            print(f"\nSkipping folder {participant} as it is not in the specified participant list.\n")
            continue

        print(f"\n####################\nAssigning labels for participant {participant}\n####################\n")
        images_folder = os.path.join(data_folder, participant, image_folder)
        game_log_folder = os.path.join(data_folder, participant, logs_folder, "CSV")

        # iterate over all logs for all games and concatenate them into one large pandas dataframe
        all_game_logs = merge_csv_logs(game_log_folder)

        # remove the tutorial rows
        all_logs = all_game_logs[all_game_logs.difficulty != 'tutorial']
        # remove the centaur game (== "3D-Game-1") and Asteroids (== "2D-Game-2") from the dataframe as well!
        all_logs = all_logs[~all_logs.gameid.isin(["3D-Game-1", "2D-Game-2"])]

        # and sort the df based on the timestamp column in ascending order
        all_logs_sorted = all_logs.sort_values('timestamp')
        # print(all_logs_sorted['timestamp'].equals(all_game_logs['timestamp']))

        df_start_end = all_logs_sorted[all_logs_sorted['event_type'].isin(start_events + end_events)]
        # remove unnecessary columns
        df_start_end_small = df_start_end.drop(['state_id', 'seed', 'score', 'args'], axis=1)

        # some sanity checks and assertions to make sure the logs are correct
        perform_checks(df_start_end_small)

        # extract all image timestamps from the image names and save them with the image index position
        all_images = os.listdir(images_folder)
        img_index_dict = {}
        img_file_dict = {}
        for idx, image_file in enumerate(all_images):
            timestamp = int(get_timestamp_from_image(image_file))
            img_index_dict[timestamp] = idx
            img_file_dict[timestamp] = image_file

        # convert the keys from string to int
        # img_index_dict = {int(key): value for key, value in img_index_dict.items()}

        # sort the image timestamp as well (even though the order should be already correct right now but whatever)
        # see https://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key#comment89671526_9001529
        sorted_img_index_dict = dict(sorted(img_index_dict.items()))
        sorted_img_file_dict = dict(sorted(img_file_dict.items()))

        result_dict = find_image_event_indexes(df_start_end_small, all_images, sorted_img_index_dict,
                                               sorted_img_file_dict, smallest_fps_val)
        participant_dict[participant] = result_dict

        # check if any difficulty level of this participant has fewer entries than the current minimum
        for image_list in result_dict.values():
            if smallest_difficulty_list_len is None or len(image_list) < smallest_difficulty_list_len:
                smallest_difficulty_list_len = len(image_list)

    print(f"List with fewest elements has {smallest_difficulty_list_len} items")
    for participant_name, difficulty_dict in participant_dict.items():
        # cut off the last elements of every list that is longer than the minimum, so we have the exact same amount of
        # image elements per difficulty (and therefore, per participant as well)
        for difficulty, image_list in difficulty_dict.items():
            len_diff = len(image_list) - smallest_difficulty_list_len
            if len_diff > 0:
                del image_list[-len_diff:]

        print(f"\n####################Creating labeled images file for {participant_name} ...\n####################")
        split_image_folder(difficulty_dict, participant_name)


if __name__ == "__main__":
    assign_load(participant_list=[])
