#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import sys
import pandas as pd
from collections import defaultdict
from post_processing.assign_load_classes import get_timestamp_from_image, find_closest_value
from post_processing.post_processing_constants import evaluation_download_folder, image_folder
from post_processing.extract_downloaded_data import get_smallest_fps


# for the evaluation study we had to manually check via the recorded screenshots when a participant started and
# finished a certain difficulty level (simply taking the timestamp in the name of the nearest screenshot),
# so the process can't be automated unfortunately.
# We always took the timestamp of the first image in-game and the last image in-game found for this difficulty.
# TODO: would it be good if the difficulty levels were in the same order as in training ??
participant_timestamp_dict = {
    "participant_1": {
        "game": "Tetris",
        "difficulty_order": ["medium", "easy", "hard"],
        "timestamps": {
            "easy_start": 1632163769969.1873,  # next: 1632163775667.6677
            "easy_end": 1632164374900.1838,
            "hard_start": 1632164489104.361,  # next: 1632164494851.8389
            "hard_end": 1632165089862.9688,
            "medium_start": 1632162947339.7017,  # next: 1632162953008.945
            "medium_end": 1632163558451.807
        }
    },
    "participant_2": {
        "game": "Age_of_Empires_II",
        "difficulty_order": ["hard", "easy", "medium"],
        "timestamps": {
            "easy_start": 1632153302916.6099,  # next: 1632153308817.8572
            "easy_end": 1632153836313.0532,
            "hard_start": 1632152455619.088,  # next: 1632152461457.088
            "hard_end": 1632153060445.0908,
            "medium_start": 1632153979122.7034,   # next: 1632153985061.2039
            "medium_end": 1632154579902.5437
        }
    }
}


def normalize_images_per_participant(sorted_img_file_dict, current_timestamp, start_timestamp, time_dist=100):
    time_diff_overall = int(current_timestamp - start_timestamp)
    time_diff_minutes = int(time_diff_overall/(1000*60))
    print(f"Time difference between start and end: {time_diff_overall/1000} seconds (~ {time_diff_minutes} minutes)")

    image_list = []

    for t in range(0, time_diff_overall, time_dist):
        new_time = start_timestamp + t
        image = find_closest_value(sorted_img_file_dict, new_time)
        image_list.append(image)

    return image_list


def generate_difficulty_label_csv(result_dict, participant_folder):
    """
    Param `result_dict` contains a list for each of the 3 difficulty levels which contains all images from this
    participant that were recorded during a game with the corresponding difficulty:
    ```
    {
        'hard': [capture__1223445.34.png, capture__122673445.89.png, ...],
        'medium': [...],
        'easy': [...],
    }
    ```
    """

    print("Number images 'easy':", len(result_dict['easy']))
    print("Number images 'hard':", len(result_dict['hard']))
    print("Number images 'medium':", len(result_dict['medium']))

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
        lambda name: f"{os.path.join(evaluation_download_folder, participant_folder, image_folder, name)}"
    )
    df_exploded.to_csv(os.path.join(evaluation_download_folder, participant_folder, "labeled_images.csv"), index=False)


def assign_labels(participant_list=list[str]):
    smallest_fps_val = get_smallest_fps(evaluation_study_mode=True)
    time_dist = 70  # TODO fixed at 70 as this should be the same as in training!

    # TODO 4284 as this was the actual value in the data collection study; needs to be the same for prediction with
    #  our pre-trained model!
    smallest_difficulty_list_len = 4284  # length of the list with the least entries over all participants

    participant_dict = defaultdict(dict)

    for participant in os.listdir(evaluation_download_folder):
        # if specific participants are given, skip the others
        if len(participant_list) > 0 and participant not in participant_list:
            print(f"\nSkipping folder {participant} as it is not in the specified participant list.\n")
            continue

        print(f"\n####################\nEvaluation Study Data: Assigning labels for"
              f" {participant}\n####################\n")
        images_folder = os.path.join(evaluation_download_folder, participant, image_folder)

        # extract all image timestamps from the image names and save them with the image index position
        all_images = os.listdir(images_folder)
        img_index_dict = {}
        img_file_dict = {}
        for idx, image_file in enumerate(all_images):
            timestamp = int(get_timestamp_from_image(image_file))
            img_index_dict[timestamp] = idx
            img_file_dict[timestamp] = image_file

        # sort the image timestamp as well (even though the order should be already correct right now but whatever)
        # see https://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key#comment89671526_9001529
        sorted_img_index_dict = dict(sorted(img_index_dict.items()))
        sorted_img_file_dict = dict(sorted(img_file_dict.items()))

        # get the participant data
        participant_entry = participant_timestamp_dict[participant]
        # game_played = participant_entry["game"]
        # difficulty_order = participant_entry["difficulty_order"]
        timestamps = participant_entry["timestamps"]  # this is a dict itself

        result_dict = defaultdict(list)
        start_pos_timestamp = None
        start_pos = None
        end_pos = None

        for game_event, timestamp in timestamps.items():
            if game_event.endswith("_start"):
                start_pos = find_closest_value(sorted_img_index_dict, timestamp)
                start_pos_timestamp = timestamp
            elif game_event.endswith("_end"):
                end_pos = find_closest_value(sorted_img_index_dict, timestamp)
                if start_pos and end_pos:
                    # get the difficulty level from the game event
                    difficulty = game_event.split("_")[0]
                    print(f"Difficulty: \"{difficulty}\":")
                    new_image_slice = normalize_images_per_participant(sorted_img_file_dict, timestamp,
                                                                       start_pos_timestamp, time_dist)

                    print(f"{len(new_image_slice)} images for {participant} & \"{difficulty}\" after normalization\n")
                    result_dict[difficulty].extend(new_image_slice)
                elif end_pos < start_pos:
                    sys.stderr.write("Something went horribly wrong... End pos should always be larger than start pos!")
            else:
                sys.stderr.write("Encountered unknown GameEvent while trying to assign difficulty labels: ", game_event)

        participant_dict[participant] = result_dict


    for participant_name, difficulty_dict in participant_dict.items():
        # cut off the last elements of every list that is longer than the minimum, so we have the exact same amount of
        # image elements per difficulty (and therefore, per participant as well)
        for difficulty, image_list in difficulty_dict.items():
            len_diff = len(image_list) - smallest_difficulty_list_len
            if len_diff > 0:
                del image_list[-len_diff:]  # remove the elements at the end if the list is too long

        print(f"\n####################\nEvaluation Study Data: Creating labeled images file for {participant_name} "
              f"...\n####################")
        generate_difficulty_label_csv(difficulty_dict, participant_name)


if __name__ == "__main__":
    assign_labels(participant_list=[])
