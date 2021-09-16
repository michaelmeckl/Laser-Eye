import os
import psutil
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from machine_learning_predictor.machine_learning_constants import results_folder
from machine_learning_predictor.ml_utils import show_result_plot


class DifficultyImageClassifier:
    # see https://keras.io/examples/vision/conv_lstm/

    def __init__(self, train_generator, val_generator, num_classes, num_epochs):
        self.n_classes = num_classes
        self.n_epochs = num_epochs

        self.train_generator = train_generator
        self.validation_generator = val_generator

        # check the maximum number of available cores
        cpu_count_available = len(psutil.Process().cpu_affinity()),  # number of usable cpus by this process
        print("CPU Count available", cpu_count_available)
        self.num_workers = cpu_count_available[0] if cpu_count_available[0] else 1

    def build_model(self, input_shape):
        print("\nBuilding ConvLSTM2D - NN ...")

        # Model architecture based on https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7
        image_input = tf.keras.Input(shape=input_shape[1:], name='image_input')

        first_ConvLSTM = tf.keras.layers.ConvLSTM2D(filters=20, kernel_size=(3, 3), recurrent_activation='hard_sigmoid',
                                                    activation='tanh', padding='same',
                                                    return_sequences=True)(image_input)
        first_BatchNormalization = tf.keras.layers.BatchNormalization()(first_ConvLSTM)
        first_Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(first_BatchNormalization)

        second_ConvLSTM = tf.keras.layers.ConvLSTM2D(filters=20, kernel_size=(3, 3), padding='same',
                                                     return_sequences=True)(first_Pooling)
        second_BatchNormalization = tf.keras.layers.BatchNormalization()(second_ConvLSTM)
        second_Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), padding='same')(second_BatchNormalization)

        third_ConvLSTM = tf.keras.layers.ConvLSTM2D(filters=10, kernel_size=(3, 3), stateful=False,
                                                    kernel_initializer='random_uniform',
                                                    padding='same', return_sequences=True)(second_Pooling)
        third_Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')(third_ConvLSTM)
        time_distributed_flat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(third_Pooling)

        # lstm_layer = tf.keras.layers.LSTM(256, stateful=True, return_sequences=True, dropout=0.5)(
        # time_distributed_flat_layer)
        first_Dense = tf.keras.layers.Dense(256)(time_distributed_flat_layer)
        dropout = tf.keras.layers.Dropout(0.2)(first_Dense)
        flat_layer = tf.keras.layers.Flatten()(dropout)
        target = tf.keras.layers.Dense(self.n_classes, activation='softmax')(flat_layer)

        self.model = tf.keras.Model(inputs=image_input, outputs=target, name='ConvLSTM2D_Model')
        self.model.summary()

        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["categorical_accuracy"])
        return self.model

    def train_classifier(self):
        checkpoint_path = os.path.join(results_folder, "checkpoints_time_series_convlstm",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)

        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        history = self.model.fit(self.train_generator,
                                 validation_data=self.validation_generator,
                                 # shuffle=False,
                                 use_multiprocessing=False,
                                 workers=self.num_workers,
                                 epochs=self.n_epochs,
                                 initial_epoch=0,
                                 callbacks=[checkpoint_callback, lr_callback],
                                 verbose=1)

        model_path = os.path.join(results_folder, "Difficulty-CNN-TimeSeries-ConvLSTM.h5")
        self.model.save(model_path)

        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_time_series.png",
                         show=False)

        return history

    def evaluate_classifier(self):
        val_loss, val_acc = self.model.evaluate(self.validation_generator, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def predict(self, img_batch, correct_labels):
        predictions = self.model.predict(img_batch)
        return predictions
