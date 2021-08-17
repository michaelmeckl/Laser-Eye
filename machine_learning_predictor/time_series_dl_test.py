#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense


def build_model_alternative(input_shape):
    filters = 4
    model = tf.keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    # TODO reshape before to (batchsize, inputsize, widht, height, channels) ???
    model.add(keras.layers.ConvLSTM2D(filters, (3, 3), padding='same', return_sequences=True))  # TODO not working!
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ConvLSTM2D(filters * 2, (3, 3), padding='same', return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ConvLSTM2D(filters, (3, 3), padding='same', return_sequences=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(3, (3, 3), padding='same'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_acc_metric = tf.keras.metrics.MeanAbsoluteError()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    return model


def build_model():
    # model from https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=5, kernel_size=3, activation="relu"), batch_input_shape=(24, None, 24, 1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, stateful=True, return_sequences=True))
    model.add(LSTM(10, stateful=True))
    model.add(Dense(24))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.CategoricalAccuracy()])


def main():
    build_model()


if __name__ == "__main__":
    main()
