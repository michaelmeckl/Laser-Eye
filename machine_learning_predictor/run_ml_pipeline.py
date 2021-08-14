#!/usr/bin/python3
# -*- coding:utf-8 -*-

from preprocess_image_data import start_preprocessing
# from cnn_test import train_model


def main():
    # preprocess data
    start_preprocessing()
    # train a machine learning model
    # train_model()  # TODO doesn't work right now; file needs refactoring to work in pipeline


if __name__ == "__main__":
    main()
