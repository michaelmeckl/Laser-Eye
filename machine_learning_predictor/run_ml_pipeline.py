#!/usr/bin/python3
# -*- coding:utf-8 -*-

# from preprocess_image_data_batch_version import start_preprocessing
from preprocess_image_data import start_preprocessing
from cnn_test import train_model


def main():
    # preprocess data
    start_preprocessing()
    # train a machine learning model
    train_model()


if __name__ == "__main__":
    main()
