#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import pathlib
import sys
import pysftp
from paramiko.ssh_exception import SSHException
from tracking.TrackingLogger import get_server_credentials
from post_processing.post_processing_constants import download_folder


def download_data_from_server(folder_names=list[str]):
    credentials = get_server_credentials(pathlib.Path(__file__).parent.parent / "sftp_credentials.properties")
    if credentials is None:
        sys.stderr.write("Reading sftp server credentials didn't work! Terminating program...")
        sys.exit(1)

    hostname = credentials["sftp_hostname"]
    username = credentials["sftp_username"]
    password = credentials["sftp_password"]
    port = credentials.getint("sftp_port")
    # NOTE: setting hostkeys to None is highly discouraged as we lose any protection
    # against man-in-the-middle attacks; however, it is the easiest solution and for now it should be fine;
    # see https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp for correct solution
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    try:
        sftp = pysftp.Connection(host=hostname, username=username, password=password, port=port, cnopts=cnopts)
    except SSHException as e:
        sys.stderr.write(f"Could not connect to server! Error message: {e}")
        sys.exit(1)

    # create the local folder to put the downloaded files into
    if not os.path.exists(download_folder):
        os.mkdir(download_folder)

    # iterate over all folders on the server and download all zipped image files per participant
    root_dir = "home"
    i = 0
    for directory in sftp.listdir(root_dir):
        if not directory.startswith("tracking_data__"):
            # we only want the folders starting with "tracking_data__"
            continue

        # we update at the start so the first folder will count as 1 and we count even if it is not in the
        # given folder names
        i += 1

        # check if specified directory names were given; if so only download these from the server
        if len(folder_names) > 0 and directory not in folder_names:
            print(f"\nSkipping folder {directory} as it is not in the specified folder names.\n")
            continue

        # important to not use os.path.join as this would use the local machine's path separator which might not be
        # the same as on the server!
        dir_path = f"{root_dir}/{directory}"
        images_path = f"{dir_path}/images"

        # some checks
        if not sftp.isdir(images_path):
            sys.stderr.write(f"\nNo images subfolder found in {root_dir}/{directory}!\n")
        if len(sftp.listdir(images_path)) == 0:
            sys.stderr.write(f"\nImages subfolder found in {root_dir}/{directory} is empty!\n")

        # check how many participants we have downloaded already to determine the number of the next participant
        num_local_participants = len(os.listdir(download_folder))
        next_participant_num = num_local_participants + 1

        local_folder = os.path.join(download_folder, f"participant_{next_participant_num}")
        if os.path.exists(local_folder):
            print(f"A folder with the name 'participant_{next_participant_num}' does already exist!")
            answer = input("Do you want to overwrite it with the data on the server? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                print(f"\nOverwriting participant_{next_participant_num}...\n")
            else:
                print(f"\nSkipping folder {i} ('{directory}') on the server.\n")
                continue

        local_images_folder = os.path.join(local_folder, "images")
        if not os.path.exists(local_images_folder):
            os.makedirs(local_images_folder)

        print(f"Downloading folder:  {dir_path}")
        print("     Downloading zipped images. This might take a few minutes ...")
        # download images folder separately
        sftp.get_d(images_path, os.path.join(local_images_folder), preserve_mtime=True)
        print("     Finished downloading images. Downloading the other logs ...")
        # then download the other files in the folder
        for file in sftp.listdir(dir_path):
            if sftp.isfile(f"{dir_path}/{file}"):
                sftp.get(f"{dir_path}/{file}", os.path.join(local_folder, file))


if __name__ == "__main__":
    # change this to an empty list to download all participants
    specified_participants = ["tracking_data__1627402551434.5142", "tracking_data__1627411166841.7903"]
    download_data_from_server(specified_participants)  # download the participant data from our sftp server
