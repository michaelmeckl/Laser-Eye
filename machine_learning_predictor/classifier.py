import os
import numpy as np
import psutil
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from machine_learning_predictor.machine_learning_constants import results_folder, NEW_IMAGE_SIZE
from machine_learning_predictor.ml_utils import show_result_plot


# TODO (val) loss on the first epoch should always be: -ln(1/n) with n = number_of_classes = 3
#  -> 1.0986 here


# noinspection PyAttributeOutsideInit
class DifficultyImageClassifier:
    """
    Custom CNN for predicting the difficulty level with images of a user's face.
    """

    def __init__(self, train_generator, val_generator, num_classes, num_epochs=30):
        self.n_classes = num_classes
        self.n_epochs = num_epochs

        self.train_generator = train_generator
        self.validation_generator = val_generator

        # check the maximum number of available cores
        cpu_count_available = len(psutil.Process().cpu_affinity()),  # number of usable cpus by this process
        print("CPU Count available", cpu_count_available)
        self.num_workers = cpu_count_available[0] if cpu_count_available[0] else 1

    # TODO try VGG-16 and check if it changes something if a different architecture is used, see
    #  https://medium.com/deep-learning-with-keras/tf-data-build-efficient-tensorflow-input-pipelines-for-image-datasets-47010d2e4330

    # TODO: use keras.Tuner or Optuna for hyperparameter search!
    def build_model(self, input_shape: tuple) -> tf.keras.Model:
        # TODO test with different architectures:
        self.sequential_model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # TODO use padding='same' ?
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

                tf.keras.layers.Flatten(),
                # units in the last layer should be a power of two
                tf.keras.layers.Dense(units=512, activation="relu"),
                # tf.keras.layers.Dropout(0.3),  # TODO add BatchNormalization() Layers instead of Dropout ?

                # units must be the number of classes -> we want a vector that looks like this: [0.2, 0.5, 0.3]
                tf.keras.layers.Dense(units=self.n_classes, activation="softmax")
                # use softmax for multi-class classification, see
                # https://medium.com/deep-learning-with-keras/how-to-solve-classification-problems-in-deep-learning-with-tensorflow-keras-6e39c5b09501
            ]
        )

        self.sequential_model.summary()
        # for the choice of last layer, activation and loss functions see
        # https://medium.com/deep-learning-with-keras/which-activation-loss-functions-in-multi-class-clasification-4cd599e4e61f
        self.sequential_model.compile(optimizer="adam",
                                      loss="categorical_crossentropy",
                                      metrics=["categorical_accuracy"])

        # TODO use SparseCategoricalCrossentropy and accuracy as metrics?
        # self.sequential_model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(
        # from_logits=True), metrics=['accuracy'])

        tf.keras.utils.plot_model(self.sequential_model, "sequential_model_graph.png")
        return self.sequential_model

    def build_model_functional_version(self, input_shape: tuple) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        flatten = tf.keras.layers.Flatten()(pool3)
        dense1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
        dropout = tf.keras.layers.Dropout(0.5)(dense1)
        # units must be the number of classes -> we want a vector that looks like this: [0.2, 0.5, 0.3]
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(dropout)

        self.model = tf.keras.Model(inputs=inputs, outputs=output, name="functional_model_version")
        print(self.model.summary())

        tf.keras.utils.plot_model(self.model, to_file="functional_model_graph.png")  # needs graphviz and pydot to work!
        tf.keras.utils.plot_model(self.model, "functional_model_graph_with_shape_info.png", show_shapes=True)

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
        return self.model

    def build_and_train_multi_input_model(self, train, val, sequence_length, input_shape: tuple):
        # Idea based on https://stackoverflow.com/questions/53020898/multiple-input-cnn-for-images
        branches = []
        for i in range(sequence_length):
            inputs = tf.keras.Input(shape=input_shape)
            conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
            pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
            pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
            flatten = tf.keras.layers.Flatten()(pool2)

            model = tf.keras.Model(inputs=inputs, outputs=flatten)
            branches.append(model)

        combinedInput = tf.keras.layers.concatenate([branch.output for branch in branches])

        dense1 = tf.keras.layers.Dense(128, activation='relu')(combinedInput)
        dropout = tf.keras.layers.Dropout(0.5)(dense1)
        # units must be the number of classes -> we want a vector that looks like this: [0.2, 0.5, 0.3]
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(dropout)

        self.model = tf.keras.Model(inputs=[branch.input for branch in branches], outputs=output)
        # print(self.model.summary())
        # tf.keras.utils.plot_model(self.model, to_file="multi_input_model_graph.png")

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        all_train_batches = []
        all_train_labels = []
        all_val_labels = []
        all_val_batches = []

        # for i in range(train.n, 10):
        for i in range(10):
            batch_train, batch_train_labels = train[i]
            batch_val, batch_val_labels = val[i]

            all_train_batches.append(batch_train)
            all_val_batches.append(batch_val)
            all_train_labels.append(batch_train_labels)
            all_val_labels.append(batch_val_labels)

        self.model.fit(x=all_train_batches, y=all_train_labels, validation_data=(all_val_batches, all_val_labels),
                       use_multiprocessing=False,
                       # workers=self.num_workers,
                       epochs=self.n_epochs,
                       verbose=1)

    def log_custom_generator_info_before(self, batch, logs):
        print("\nBefore batch: ", batch, flush=True)
        print("Index list len:", len(self.train_generator.indices_list), flush=True)

    def log_custom_generator_info_after(self, batch, logs):
        print("\nAfter batch: ", batch, flush=True)
        print("Index list len:", len(self.train_generator.indices_list), flush=True)

    def train_classifier(self):
        self.step_size_train = self.train_generator.n // (self.train_generator.sequence_length *
                                                          self.train_generator.batch_size)
        self.step_size_val = self.validation_generator.n // (self.validation_generator.sequence_length *
                                                             self.validation_generator.batch_size)

        model_name = "Difficulty-CNN-Model-Generator.h5"
        model_path = os.path.join(results_folder, model_name)

        checkpoint_path = os.path.join(results_folder, "checkpoints_generator",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        # log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

        # define a custom callback
        custom_callback = tf.keras.callbacks.LambdaCallback(
            on_batch_begin=self.log_custom_generator_info_before, on_batch_end=self.log_custom_generator_info_after)

        history = self.sequential_model.fit(self.train_generator,
                                            validation_data=self.validation_generator,
                                            shuffle=False,   # VERY IMPORTANT even when used with custom data generator!
                                            # see https://github.com/keras-team/keras/issues/12082#issuecomment-455877627
                                            use_multiprocessing=False,
                                            workers=self.num_workers,
                                            epochs=self.n_epochs,
                                            callbacks=[checkpoint_callback, lr_callback],  # custom_callback],
                                            verbose=1)

        self.sequential_model.save(model_path)
        show_result_plot(history, self.n_epochs, metric="categorical_accuracy",
                         output_name="train_history_custom_generator.png")

        return history

    def train_classifier_dataset_version(self, train_ds, val_ds):
        model_name = "Difficulty-CNN-Model-Dataset.h5"
        model_path = os.path.join(results_folder, model_name)

        checkpoint_path = os.path.join(results_folder, "checkpoints_dataset",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        # early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max',
        #                                                  restore_best_weights=True, patience=5, verbose=1)

        custom_callback = tf.keras.callbacks.LambdaCallback(
            on_batch_begin=self.log_custom_generator_info_before, on_batch_end=self.log_custom_generator_info_after)

        history = self.sequential_model.fit(train_ds,
                                            validation_data=val_ds,
                                            shuffle=False,
                                            epochs=self.n_epochs,
                                            callbacks=[checkpoint_callback, lr_callback],  # custom_callback],
                                            verbose=1)

        self.sequential_model.save(model_path)
        show_result_plot(history, self.n_epochs, metric="categorical_accuracy",
                         output_name="train_history_custom_dataset.png")

        return history

    def evaluate_classifier(self):
        val_loss, val_acc = self.sequential_model.evaluate(self.validation_generator,
                                                           steps=self.step_size_val,
                                                           verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def evaluate_classifier_dataset_version(self, val_ds):
        val_loss, val_acc = self.sequential_model.evaluate(val_ds, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def predict(self, test_ds):
        predict_image_path = test_ds[0]
        img = tf.keras.preprocessing.image.load_img(
            predict_image_path, target_size=NEW_IMAGE_SIZE  # TODO use the same resizing as for training!
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.sequential_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print("Prediction:", predictions)
        print("Score:", score)

        category = test_ds.class_names[np.argmax(score)]
        print(f"This image most likely belongs to {category} with a {100 * np.max(score):.2f} percent confidence.")
