from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class CustomNetworkWrapper:

    def __init__(self, no_of_features=0, no_of_classes=90, model_identifier='FeedForward', given_model=None):

        # Base initialization
        optimizer = 'adam'
        loss = 'categorical_crossentropy',
        # TODO: This breaks the self.model.evaluate function atm. Fix it!
        metrics = [tfk.metrics.Accuracy(), tfk.metrics.Recall(), tfk.metrics.Precision()]

        if model_identifier == 'GivenModel':
            if given_model is None:
                print("The CustomNetworkWrapper was instructed to initialize with a given model but none was provided.")
                print("A critical error is imminent!")
            else:
                self.model = given_model

        elif model_identifier == 'Reuters-FeedForward':
            if no_of_features <= 0:
                print("The CustomNetworkWrapper expects a positive number of features for the reuters dataset.")
                print("A critical error is imminent!")
            middle_size = np.round(np.sqrt(no_of_features * 300 * no_of_classes))
            self.model = tfk.Sequential([
                tfk.layers.Input(shape=(no_of_features, 300)),
                tfk.layers.Flatten(input_shape=(no_of_features, 300)),
                # 300 = dimensionality of embedding
                tfk.layers.Dense(300 * no_of_features,
                                 activation=tf.nn.relu),
                # middle layer is chosen so the downscaling factor is constant
                tfk.layers.Dense(middle_size,
                                 activation=tf.nn.relu),
                # no_of_classes = number of categories
                tfk.layers.Dense(no_of_classes,
                                 activation=tf.nn.softmax)
            ])

        elif model_identifier == 'Reuters-CNN':
            if no_of_features <= 0:
                print("The CustomNetworkWrapper expects a positive number of features for the reuters dataset.")
                print("A critical error is imminent!")
            kernel_width = 3
            no_of_filters = no_of_features - kernel_width
            self.model = tfk.Sequential([
                tfk.layers.Conv2D(filters=1,
                                  kernel_size=(kernel_width, 300),
                                  activation='relu',
                                  input_shape=(no_of_features, 300, 1)),
                tfk.layers.Flatten(input_shape=(no_of_filters, 1, 1)),
                tfk.layers.Dense(units=no_of_classes,
                                 input_shape=(no_of_filters,),
                                 activation=tf.nn.softmax)
            ])

        elif model_identifier == 'MNIST-Lenet-FC':
            self.model = tfk.Sequential([
                tfk.layers.Input(shape=(28, 28)),
                tfk.layers.Flatten(input_shape=(28, 28)),
                # Implicit Activation is linear
                # TODO: Find out whether the lottery ticket paper uses a more sophisticated activation or not
                tfk.layers.Dense(units=300,
                                 activation='relu'),
                tfk.layers.Dense(units=100,
                                 activation='relu'),
                tfk.layers.Dense(units=10,
                                 activation=tf.nn.softmax)
            ])
            optimizer = tfk.optimizers.Adam(learning_rate=1.2*1e-03)
            # loss = tfk.losses.mean_squared_error
            # Not supported by self.evaluate_model()  yet
            metrics = [tfk.metrics.Accuracy(), tfk.metrics.Recall(), tfk.metrics.Precision()]

        else:
            print("CustomNetworkHandler does not handle a model type called: ", model_identifier)
            print("Please use one of the following names:")
            print("'Given-Model'")
            print("'Reuters-FeedForward', 'Reuters-CNN'")

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        self.model.summary()
        print(self.model.get_config())

    def train_model(self, datapoints, epochs, batch_size=32, verbosity=1):
        history = self.model.fit(datapoints[0],
                                 datapoints[1],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbosity)
        return history

    def train_model_with_validation(self, datapoints, validation_datapoints,
                                    epochs, batch_size=32, verbosity=1):
        history = self.model.fit(datapoints[0],
                                 datapoints[1],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbosity,
                                 validation_data=validation_datapoints)
        return history

    def save_model_as_folder(self, foldername):
        tfk.models.save_model(
            self.model,
            'SavedModels/' + foldername)

        return

    def evaluate_model(self, datapoints):
        # TODO: extend
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

