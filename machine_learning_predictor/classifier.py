import os
from typing import Optional
import psutil
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from machine_learning_predictor.machine_learning_constants import results_folder
from machine_learning_predictor.ml_utils import show_result_plot


# noinspection PyAttributeOutsideInit
class DifficultyImageClassifier:
    """
    Custom CNN for predicting the difficulty level with images of a user's face.
    """

    def __init__(self, train_generator, val_generator, num_classes, num_epochs=20):
        self.n_classes = num_classes
        self.n_epochs = num_epochs

        self.train_generator = train_generator
        self.validation_generator = val_generator

        self.step_size_train = train_generator.n // train_generator.batch_size
        self.step_size_val = val_generator.n // val_generator.batch_size

        self.model_name = "Difficulty-CNN-Model.h5"
        self.model_path = os.path.join(results_folder, self.model_name)

        self.checkpoint_path = os.path.join(results_folder, "checkpoints",
                                            "checkpoint-improvement-{epoch:02d}-{categorical_accuracy:.3f}.h5")

    def build_model(self, input_shape: tuple[Optional[int], int, int, int]) -> tf.keras.Model:
        # TODO test with different architectures:
        self.sequential_model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),

                # TODO use padding='same' ?
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),

                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),

                tf.keras.layers.Flatten(),
                # units in the last layer should be a power of two
                tf.keras.layers.Dense(units=1024, activation="relu"),  # TODO test with fewer, e.g. 128?
                tf.keras.layers.Dropout(0.5),

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
        # TODO
        # self.sequential_model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(
        # from_logits=True), metrics=['accuracy'])

        return self.sequential_model

    def build_model_functional_version(self, input_shape: tuple[Optional[int], int, int, int]) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape)
        output = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(output)
        output = tf.keras.layers.Dropout(0.25)(output)

        output = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(output)
        output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(output)
        output = tf.keras.layers.Dropout(0.25)(output)

        output = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(output)
        output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(output)
        output = tf.keras.layers.Dropout(0.25)(output)

        output = tf.keras.layers.Flatten()(output),
        # units in the last layer should be a power of two
        output = tf.keras.layers.Dense(units=1024, activation="relu")(output),  # TODO not working at this point!
        output = tf.keras.layers.Dropout(0.5)(output),

        # units must be the number of classes -> we want a vector that looks like this: [0.2, 0.5, 0.3]
        output = tf.keras.layers.Dense(units=self.n_classes, activation="softmax")(output)
        self.functional_model = tf.keras.Model(inputs, output)

        return self.functional_model

    def train_classifier(self):
        # check the maximum number of available cores
        cpu_count_available = len(psutil.Process().cpu_affinity()),  # number of usable cpus by this process
        print("CPU Count available", cpu_count_available)
        num_workers = cpu_count_available[0] if cpu_count_available[0] else 1

        # save checkpoints
        checkpoint_callback = ModelCheckpoint(self.checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        # history = self.sequential_model.fit_generator(generator=self.train_generator,
        history = self.sequential_model.fit(self.train_generator,
                                            steps_per_epoch=self.step_size_train,
                                            validation_data=self.validation_generator,
                                            validation_steps=self.step_size_val,
                                            use_multiprocessing=False,
                                            workers=num_workers,
                                            epochs=self.n_epochs,
                                            callbacks=[checkpoint_callback, lr_callback],
                                            verbose=1)

        show_result_plot(history, self.n_epochs, metric="categorical_accuracy",
                         output_name="train_history_custom_generator.png")

        self.sequential_model.save(self.model_path)
        return history

    def evaluate_classifier(self):
        val_loss, val_acc = self.sequential_model.evaluate(self.validation_generator,
                                                           steps=self.step_size_val,
                                                           verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def predict(self, test_images):
        pass
