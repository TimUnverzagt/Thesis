from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import sklearn
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


def init_submodel(kernel_size, pool_size):
    submodel = tfk.Sequential()
    submodel.add(tfk.layers.Embedding(input_dim=61188,
                                      input_length=200,
                                      output_dim=300))
    submodel.add(tfk.layers.Conv1D(filters=3,
                                   kernel_size=kernel_size))
    submodel.add(tfk.layers.AvgPool1D(pool_size=pool_size))
    submodel.add(tfk.layers.Dropout(rate=0.5))
    submodel.add(tfk.layers.GlobalAveragePooling1D())
    return submodel


class CustomNetworkWrapper:

    def __init__(self, no_of_features=0, no_of_classes=90, model_identifier='FeedForward', given_model=None,
                 summarize=False):

        # Base initialization
        optimizer = 'adam'
        loss = 'categorical_crossentropy',
        # TODO: This breaks the self.model.evaluate function atm. Fix it!
        metrics = []
        # metrics = [tfk.metrics.Accuracy(), tfk.metrics.Recall(), tfk.metrics.Precision()]

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

        elif model_identifier == 'MNIST-Lenet-FCN':
            self.model = tfk.Sequential([
                tfk.layers.Input(shape=(28, 28)),
                tfk.layers.Flatten(input_shape=(28, 28)),
                tfk.layers.Dense(units=300,
                                 activation='relu'),
                tfk.layers.Dense(units=100,
                                 activation='relu'),
                tfk.layers.Dense(units=10,
                                 activation=tf.nn.softmax)
            ])
            optimizer = tfk.optimizers.Adam(learning_rate=1.2*1e-03)

        elif model_identifier == 'Newsgroups-End2End-CNN':
            common_input = tfk.layers.Input(shape=200)

            sequentials = []
            seq_outputs = []
            # for i in range(16):
            for i in range(2):
                if i < 8:
                    sequentials.append(init_submodel(1+3*i, 2))
                    seq_outputs.append(sequentials[i](common_input))
                else:
                    sequentials.append(init_submodel(1+3*(i % 8), 7))
                    seq_outputs.append(sequentials[i](common_input))

            merged = tfk.layers.concatenate(seq_outputs)
            regularized = tfk.layers.Dropout(rate=0.5)(merged)
            output = tfk.layers.Dense(20,
                                      activation='softmax')(regularized)
            self.model = tfk.Model(inputs=[common_input],
                                   outputs=output)
            loss = 'sparse_categorical_crossentropy'

        elif model_identifier == 'CIFAR10-CNN-6':
            self.model = tfk.Sequential([
                tfk.layers.Input(shape=(32, 32, 3)),
                tfk.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu'),
                tfk.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu'),
                tfk.layers.MaxPool2D(),
                tfk.layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu'),
                tfk.layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu'),
                tfk.layers.MaxPool2D(),
                tfk.layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu'),
                tfk.layers.Conv2D(
                    filters=256,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu'),
                tfk.layers.MaxPool2D(),
                tfk.layers.Flatten(),
                tfk.layers.Dense(units=256,
                                 activation='relu'),
                tfk.layers.Dense(units=256,
                                 activation='relu'),
                tfk.layers.Dense(units=10,
                                 activation=tf.nn.softmax),
            ])
            optimizer = tfk.optimizers.Adam(learning_rate=3*1e-04)

        else:
            print("CustomNetworkHandler does not handle a model type called: ", model_identifier)
            print("Please use one of the following names:")
            print("'Given-Model'")
            print("'Reuters-FeedForward', 'Reuters-CNN'")
            print("'MNIST-Lenet-FCN'")
            print("'CIFAR10-CNN-6'")
            print("'Newsgroups-End2End-CNN'")

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        if summarize:
            self.model.summary()
        # print(self.model.get_config())

    def train_model(self, datapoints, epochs, batch_size=32, verbosity=1):
        history = self.model.fit(datapoints[0],
                                 datapoints[1],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbosity)
        return history.history

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




