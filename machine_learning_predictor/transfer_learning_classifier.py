import os
import numpy as np
import psutil
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from machine_learning_predictor.difficulty_levels import DifficultyLevels
from machine_learning_predictor.machine_learning_constants import results_folder
from machine_learning_predictor.ml_utils import show_result_plot


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

    def use_tensorhub(self, input_shape, train_dataset, val_dataset):
        hub_url = "https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3"
        encoder = hub.KerasLayer(hub_url, trainable=True)

        inputs = tf.keras.layers.Input(shape=input_shape[1:], dtype=tf.float32, name='image')
        encoder_output = encoder(dict(image=inputs))

        x = tf.keras.layers.Dense(256, activation="relu")(encoder_output)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.n_classes, activation="softmax")(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.summary()

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        history = self.model.fit(train_dataset, validation_data=val_dataset,
                                 use_multiprocessing=False,
                                 workers=self.num_workers,
                                 epochs=32,
                                 verbose=1)

        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_tensorhub.png",
                         show=True)

        val_loss, val_acc = self.model.evaluate(val_dataset, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def unfreeze_model(self, model, num_layers=20):
        # We unfreeze the top 20 layers per default while leaving BatchNorm layers frozen
        for layer in model.layers[-num_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    def train_transfer_model(self, input_shape):
        # see https://keras.io/guides/transfer_learning/
        inputs = tf.keras.Input(shape=input_shape)

        # base_model = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False)
        # base_model = tf.keras.applications.Xception(weights="imagenet", input_tensor=inputs, include_top=False)
        # base_model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet", input_tensor=inputs, include_top=False)
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet", input_tensor=inputs,
                                                                       include_top=False)
        # base_model = tf.keras.applications.vgg16.VGG16(weights="imagenet", input_tensor=inputs, include_top=False)

        base_model.trainable = False

        # x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.n_classes, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, outputs)

        # TODO add timeDistributed as well?
        # see https://stackoverflow.com/questions/61431708/transfer-learning-for-video-classification

        self.model.summary()

        # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        checkpoint_path = os.path.join(results_folder, "checkpoints_transfer_pretrained_gen_2",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)
        # adjust learning rate
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        history = self.model.fit(self.train_generator,
                                 validation_data=self.validation_generator,
                                 # shuffle=False,
                                 use_multiprocessing=False,
                                 workers=self.num_workers,
                                 callbacks=[checkpoint_callback, lr_callback],
                                 epochs=self.n_epochs,
                                 verbose=1)

        model_name = "Transfer-Learning-Model-Generator_2.h5"
        model_path = os.path.join(results_folder, model_name)
        self.model.save(model_path)
        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_transfer_2.png",
                         show=False)

        val_loss, val_acc = self.model.evaluate(self.validation_generator, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

        # do fine-tuning
        print("\nFine-tuning base model ...\n")

        self.unfreeze_model(self.model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        checkpoint_path = os.path.join(results_folder, "checkpoints_transfer_fine_tuned_gen_2",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)

        fine_tune_epochs = 15
        history = self.model.fit(self.train_generator,
                                 validation_data=self.validation_generator,
                                 # shuffle=False,
                                 use_multiprocessing=False,
                                 workers=self.num_workers,
                                 callbacks=[checkpoint_callback, lr_callback],
                                 epochs=self.n_epochs + fine_tune_epochs,
                                 initial_epoch=history.epoch[-1],  # start from the last epoch of the pretrained model
                                 verbose=1)

        model_name = "Transfer-Learning-Model-Fine-Tuned-Generator_2.h5"
        model_path = os.path.join(results_folder, model_name)
        self.model.save(model_path)
        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_transfer_fine_tuned_2.png",
                         show=False)

        val_loss, val_acc = self.model.evaluate(self.validation_generator, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def try_transfer_ds_version(self, input_shape, train_ds, val_ds):
        # see https://keras.io/guides/transfer_learning/
        inputs = tf.keras.Input(shape=input_shape)

        base_model = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet", input_tensor=inputs,
                                                                       include_top=False)
        base_model.trainable = False

        # x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.n_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        model_name = "Transfer-Learning-Model-Dataset.h5"
        model_path = os.path.join(results_folder, model_name)

        checkpoint_path = os.path.join(results_folder, "checkpoints_transfer_pretrained_ds",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

        history = model.fit(train_ds,
                            validation_data=val_ds,
                            use_multiprocessing=False,
                            workers=self.num_workers,
                            callbacks=[checkpoint_callback, lr_callback],
                            epochs=self.n_epochs,
                            verbose=1)

        model.save(model_path)
        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_transfer_ds.png",
                         show=False)

        val_loss, val_acc = model.evaluate(val_ds, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

        # do fine-tuning
        print("\nFine-tuning base model ...\n")
        self.unfreeze_model(model)
        checkpoint_path = os.path.join(results_folder, "checkpoints_transfer_fine_tuned_ds",
                                       "checkpoint-improvement-{epoch:02d}-{val_categorical_accuracy:.3f}.ckpt")
        # save checkpoints
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1,
                                              mode="max", save_best_only=True, save_weights_only=True)

        fine_tune_epochs = 15
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            use_multiprocessing=False,
                            workers=self.num_workers,
                            callbacks=[checkpoint_callback, lr_callback],
                            epochs=self.n_epochs + fine_tune_epochs,
                            initial_epoch=history.epoch[-1],  # start from the last epoch of the pretrained model
                            verbose=1)

        model_name = "Transfer-Learning-Model-Fine-Tuned-Dataset.h5"
        model_path = os.path.join(results_folder, model_name)
        model.save(model_path)
        show_result_plot(history, metric="categorical_accuracy", output_name="train_history_transfer_fine_tuned_ds.png",
                         show=False)

        val_loss, val_acc = model.evaluate(val_ds, verbose=1)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_acc * 100)

    def get_label_name_for_index_pos(self, index_pos):
        mask_array = np.zeros(self.n_classes, dtype=int)  # creates this: [0 0 0]
        mask_array[index_pos] = 1  # if index_pos was 0: [1 0 0]
        return DifficultyLevels.get_label_for_encoding(mask_array)

    def predict(self, img_batch, correct_labels):
        predictions = self.model.predict(img_batch)
        """
        for i, (prediction, correct_label) in enumerate(zip(predictions, correct_labels)):
            score = tf.nn.softmax(prediction)
            print(f"\nPrediction for sequence {i}: {prediction}\nScore: {score})")
            index = np.argmax(score)
            predicted_label = self.get_label_name_for_index_pos(index)
            print(f"Correct label is  \"{DifficultyLevels.get_label_for_encoding(correct_label)}\"")
            print(f"Predicted label was \"{predicted_label}\" with a confidence of {100 * score[index]:.2f} %")
        """
        return predictions
