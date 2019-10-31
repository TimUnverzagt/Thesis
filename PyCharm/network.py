from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class CustomNetworkWrapper:

    def __init__(self, no_of_features, no_of_classes=90, model_identifier='FeedForward', given_model=None):

        if model_identifier == 'GivenModel':
            if given_model is None:
                # TODO: Throw real exception
                print("The CustomNetworkHandler was instructed to initialize with a given model but none was provided.")
                print("A critical error is imminent!")
            else:
                self.model = given_model
        elif model_identifier == 'FeedForward':
            middle_size = np.round(np.sqrt(no_of_features * 300 * no_of_classes))
            self.model = tfk.Sequential([
                tfk.layers.Input(shape=(no_of_features, 300)),
                tfk.layers.Flatten(input_shape=(no_of_features, 300)),
                # 300 = dimensionality of embedding
                tfk.layers.Dense(300 * no_of_features, activation=tf.nn.relu),
                # middle layer is chosen so the downscaling factor is constant
                tfk.layers.Dense(middle_size, activation=tf.nn.relu),
                # no_of_classes = number of categories
                tfk.layers.Dense(no_of_classes, activation=tf.nn.sigmoid)
            ])
        elif model_identifier == 'CNN':
            kernel_width = 3
            no_of_filters = no_of_features - kernel_width
            self.model = tfk.Sequential([
                tfk.layers.Conv2D(filters=1, kernel_size=(kernel_width, 300),
                                  activation='relu', input_shape=(no_of_features, 300, 1)),
                tfk.layers.Flatten(input_shape=(no_of_filters, 1, 1)),
                tfk.layers.Dense(units=no_of_classes, input_shape=(no_of_filters,), activation=tf.nn.sigmoid)
            ])
        else:
            print("CustomNetworkHandler does not handle a model type called: ", model_identifier)
            print("Please use one of the following names: 'FeedForward', 'CNN', 'GivenModel'")

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[tfk.metrics.Recall(), tfk.metrics.Precision()])

    def train_model(self, datapoints, epochs):
        history = self.model.fit(datapoints[0],
                                 datapoints[1],
                                 batch_size=32,
                                 epochs=epochs)
        return history

    def save_model_as_folder(self, filename):
        tfk.models.save_model(
            self.model,
            'SavedModels/' + filename)

        return

    def evaluate_model(self, datapoints):
        test_loss, test_recall, test_precision = self.model.evaluate(datapoints[0], datapoints[1])

        print("Loss: ", test_loss)
        print("Recall: ", test_recall)
        print("Precision: ", test_precision)
        if not (test_precision + test_recall) == 0:
            f1_measure = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        else:
            f1_measure = 0

        print("F1: ", f1_measure)
        return

