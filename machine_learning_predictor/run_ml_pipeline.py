#!/usr/bin/python3
# -*- coding:utf-8 -*-

from preprocess_image_data import start_preprocessing
from cnn_test import main as build_model


def main():
    # preprocess data
    start_preprocessing()
    # train a machine learning model
    build_model(train=True, test=True)


if __name__ == "__main__":
    main()
