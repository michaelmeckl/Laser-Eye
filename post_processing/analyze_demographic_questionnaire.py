#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from post_processing.post_processing_constants import download_folder, logs_folder


def merge_demographic_questionnaires(participants_path):
    all_demographic_questionnaires = pd.DataFrame()
    n = 7  # we only some columns from the csv data as some of the entries contain comma-separated
    # values and can't be automatically extracted which is why the other columns where analyzed manually

    for participant in os.listdir(participants_path):
        log_folder_path = os.path.join(participants_path, participant, logs_folder)
        for element in os.listdir(log_folder_path):
            if "endquestionaire" in element:
                demographic_questionnaire_path = os.path.join(log_folder_path, element)
                questionnaire_df = pd.read_csv(demographic_questionnaire_path, encoding='utf8',
                                               usecols=range(n), skipinitialspace=True)

                # remove whitespaces at the end
                # questionnaire_df.columns = questionnaire_df.columns.str.strip()
                # cols = questionnaire_df.select_dtypes(['object']).columns
                # questionnaire_df[cols] = questionnaire_df[cols].apply(lambda x: x.str.strip())
                all_demographic_questionnaires = pd.concat([all_demographic_questionnaires, questionnaire_df])

    all_demographic_questionnaires = all_demographic_questionnaires.reset_index(drop=True)
    return all_demographic_questionnaires


def analyze_demographics():
    participants_path = download_folder
    questionnaire_df = merge_demographic_questionnaires(participants_path)

    age_column = questionnaire_df['age']
    print(f"Mean age over all participants: {np.mean(age_column):.2f} (standard deviation: {np.std(age_column):.2f})")
    print(f"Participant age ranged from {np.min(age_column)} to {np.max(age_column)}.")

    gender_female = questionnaire_df[questionnaire_df.gender == 'weiblich']
    gender_male = questionnaire_df[questionnaire_df.gender == 'männlich']
    print(f"{len(gender_female)} female participants and {len(gender_male)} male participants took part.")


if __name__ == "__main__":
    analyze_demographics()
