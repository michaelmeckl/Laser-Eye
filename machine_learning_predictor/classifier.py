import os
import numpy as np
import psutil
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, EarlyStopping
from machine_learning_predictor.machine_learning_constants import results_folder
from machine_learning_predictor.ml_utils import show_result_plot, load_saved_model


# noinspection PyAttributeOutsideInit
class DifficultyImageClassifier:
    """
    Custom CNN for predicting the difficulty level with images of a user's face.
    """

    def __init__(self, train_generator, val_generator, num_classes, num_epochs):
        self.n_classes = num_classes
        self.n_epochs = num_epochs

        self.train_generator = train_generator
        self.validation_generator = val_generator

        # check the maximum number of available cores
        cpu_count_available = len(psutil.Process().cpu_affinity()),  # number of usable cpus by this process
        print("CPU Count available", cpu_count_available)
        self.num_workers = cpu_count_available[0] if cpu_count_available[0] else 1

    def build_model(self, input_shape: tuple, img_batch) -> tf.keras.Model:
        self.sequential_model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding="same", input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
                tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                tf.keras.layers.Flatten(),
                # units in the last layer should be a power of two
                tf.keras.layers.Dense(units=512, activation="relu"),
                tf.keras.layers.Dense(units=126, activation="relu"),

                tf.keras.layers.Dropout(0.2),
                # tf.keras.layers.BatchNormalization(),

                # units must be the number of classes -> we want a vector that looks like this: [0.2, 0.5, 0.3]
                tf.keras.layers.Dense(units=self.n_classes, activation="softmax")
                # use softmax for multi-class classification, see
                # https://medium.com/deep-learning-with-keras/how-to-solve-classification-problems-in-deep-learning-with-tensorflow-keras-6e39c5b09501
            ]
        )

        self.sequential_model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # for the choice of last layer, activation and loss functions see
        # https://medium.com/deep-learning-with-keras/which-activation-loss-functions-in-multi-class-clasification-4cd599e4e61f
        self.sequential_model.compile(optimizer=opt,
                                      loss="categorical_crossentropy",
                                      metrics=["categorical_accuracy"])

        show_conv_output = False
        if show_conv_output:
            # see https://github.com/bnsreenu/python_for_microscopists/blob/master/152-visualizing_conv_layer_outputs.py

            # Understand the filters in the model
            # Let us pick the first hidden layer as the layer of interest.
            layer = self.sequential_model.layers  # Conv layers at 1, 3, 6, 8, 11, 13, 15
            filters, biases = self.sequential_model.layers[0].get_weights()
            print(layer[0].name, filters.shape)

            # plot filters
            """
            fig1 = plt.figure(figsize=(8, 12))
            columns = 8
            rows = 8
            n_filters = columns * rows
            for i in range(1, n_filters + 1):
                f = filters[:, :, :, i - 1]
                fig1 = plt.subplot(rows, columns, i)
                fig1.set_xticks([])  # Turn off axis
                fig1.set_yticks([])
                plt.imshow(f[:, :, 0], cmap='gray')  # Show only the filters from 0th channel (R)
                # ix += 1
            plt.show()
            """

            # Define a new truncated model to only include the conv layers of interest
            # conv_layer_index = [1, 3, 6, 8, 11, 13, 15]
            conv_layer_index = [0, 2, 4]  # TO define a shorter model
            outputs = [self.sequential_model.layers[i].output for i in conv_layer_index]
            model_short = tf.keras.Model(inputs=self.sequential_model.inputs, outputs=outputs)
            print(model_short.summary())

            # Generate feature output by predicting on the input image
            feature_output = model_short.predict(img_batch)

            columns = 6  # max: np.sqrt(smallest filter size), here 8*8
            rows = 6
            for j, ftr in enumerate(feature_output):
                # pos = 1
                fig = plt.figure(figsize=(12, 12))
                fig.suptitle(f"feature output - {j}")
                for i in range(1, columns * rows + 1):
                    fig = plt.subplot(rows, columns, i)
                    fig.set_xticks([])  # Turn off axis
                    fig.set_yticks([])
                    plt.imshow(ftr[0, :, :, i - 1], cmap='gray')
                    # pos += 1
                plt.savefig(f"feature output - {j}")
                plt.show()

        # tf.keras.utils.plot_model(self.sequential_model, "sequential_model_graph.png")
        return self.sequential_model

    def build_mixed_model(self, img_input_shape, eye_log_input_shape):
        img_input = tf.keras.Input(shape=img_input_shape, name="image_input")
        conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(img_input)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        flat_1 = tf.keras.layers.Flatten()(pool3)
        """
        flat_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(pool3)
        lstm_1 = tf.keras.layers.LSTM(100, activation="relu", return_sequences=False,
                                      kernel_initializer="he_normal")(flat_1)
        """

        eye_log_input = tf.keras.Input(shape=eye_log_input_shape, name="eye_data_input")
        layer1 = tf.keras.layers.Dense(64, activation='relu')(eye_log_input)
        layer2 = tf.keras.layers.Dense(128, activation='relu')(layer1)
        layer3 = tf.keras.layers.Dense(256, activation='relu')(layer2)
        flat_2 = tf.keras.layers.Flatten()(layer3)

        combined_input = tf.keras.layers.concatenate([flat_1, flat_2])

        dense1 = tf.keras.layers.Dense(256, activation='relu')(combined_input)  # TODO 512 ?
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        # units must be the number of classes -> we want a vector that looks like this: [0.2, 0.5, 0.3]
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(dropout)

        self.model = tf.keras.Model(inputs=[img_input, eye_log_input], outputs=output)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        print(self.model.summary())
        # needs graphviz and pydot to work!
        # tf.keras.utils.plot_model(self.model, "mixed_model_graph.png", show_shapes=True)

        checkpoint_path = os.path.join(results_folder, "checkpoints_mixed_data",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)

        history = self.model.fit(self.train_generator,
                                 validation_data=self.validation_generator,
                                 use_multiprocessing=False,
                                 workers=self.num_workers,
                                 epochs=self.n_epochs,
                                 callbacks=[checkpoint_callback],
                                 verbose=1)

        model_name = "Mixed-Model.h5"
        model_path = os.path.join(results_folder, model_name)
        self.model.save(model_path)

        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_mixed_model.png",
                         show=False)

        val_loss, val_acc = self.model.evaluate(self.validation_generator, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

        return history, val_acc

    def train_classifier(self):
        model_name = "Difficulty-CNN-Model-Generator.h5"
        model_path = os.path.join(results_folder, model_name)

        checkpoint_path = os.path.join(results_folder, "checkpoints_generator",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        # log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

        # define a custom callback
        # custom_callback = LambdaCallback(
        #     on_batch_begin=self.log_custom_generator_info_before, on_batch_end=self.log_custom_generator_info_after)

        history = self.sequential_model.fit(self.train_generator,
                                            validation_data=self.validation_generator,
                                            # shuffle=False, # see
                                            # https://github.com/keras-team/keras/issues/12082#issuecomment-455877627
                                            use_multiprocessing=False,
                                            workers=self.num_workers,
                                            epochs=self.n_epochs,
                                            callbacks=[checkpoint_callback, lr_callback],  # custom_callback],
                                            verbose=1)

        self.sequential_model.save(model_path)
        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_custom_generator.png",
                         show=True)

        return history

    def train_classifier_dataset_version(self, train_ds, val_ds):
        model_name = "Difficulty-CNN-Model-Dataset.h5"
        model_path = os.path.join(results_folder, model_name)

        checkpoint_path = os.path.join(results_folder, "checkpoints_dataset",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        # early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max',
        #                                                  restore_best_weights=True, patience=5, verbose=1)

        history = self.sequential_model.fit(train_ds,
                                            validation_data=val_ds,
                                            # shuffle=False,
                                            use_multiprocessing=False,
                                            workers=self.num_workers,
                                            epochs=self.n_epochs,
                                            callbacks=[checkpoint_callback, lr_callback],
                                            verbose=1)

        self.sequential_model.save(model_path)
        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_custom_dataset.png",
                         show=True)

        return history

    def evaluate_classifier(self):
        val_loss, val_acc = self.sequential_model.evaluate(self.validation_generator,
                                                           # steps=self.step_size_val,
                                                           verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def evaluate_classifier_dataset_version(self, val_ds):
        val_loss, val_acc = self.sequential_model.evaluate(val_ds, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def predict_test_generator(self, test_gen):
        predictions = self.sequential_model.predict(test_gen)
        score = tf.nn.softmax(predictions[0])  # take the softmax over all predictions ([0] because it's nested)
        # print("Predictions:\n", predictions)
        print(f"Confidence: {100 * np.max(score):.2f} %")

        # evaluate on the test set as well
        test_loss, test_acc = self.sequential_model.evaluate(test_gen, verbose=1)
        print("Test accuracy: ", test_acc * 100)

        # load latest (i.e. the best) checkpoint
        loaded_model = load_saved_model(model_name="Difficulty-CNN-Model-Generator.h5")  # re-create the model first!
        checkpoint_folder = os.path.join(results_folder, "checkpoints_generator")  # "checkpoints_dataset"

        latest = tf.train.latest_checkpoint(checkpoint_folder)
        loaded_model.load_weights(latest)
        # and re-evaluate the model
        loss, acc = loaded_model.evaluate(test_gen, verbose=1)
        print(f"Accuracy with restored model weights: {100 * acc:5.2f}%")
