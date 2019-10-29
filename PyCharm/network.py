from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class CustomNetworkHandler:

    def __init__(self, target_doc_len, no_of_classes=90, model_name='sandbox'):

        # Hack to prevent a specific error with cudNN
        # https://github.com/tensorflow/tensorflow/issues/24828
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

        if model_name == 'sandbox':
            middle_size = np.round(np.sqrt(target_doc_len * 300 * no_of_classes))
            self.model = tfk.Sequential([
                tfk.layers.Input(shape=(target_doc_len, 300)),
                tfk.layers.Flatten(input_shape=(target_doc_len, 300)),
                # 300 = dimensionality of embedding
                tfk.layers.Dense(300 * target_doc_len, activation=tf.nn.relu),
                # middle layer is chosen so the downscaling factor is constant
                tfk.layers.Dense(middle_size, activation=tf.nn.relu),
                # no_of_classes = number of categories
                tfk.layers.Dense(no_of_classes, activation=tf.nn.sigmoid)
            ])
        elif model_name == 'CNN':
            kernel_width = 3
            no_of_filters = target_doc_len - kernel_width
            self.model = tfk.Sequential([
                tfk.layers.Conv2D(filters=1, kernel_size=(kernel_width, 300),
                                  activation='relu', input_shape=(target_doc_len, 300, 1)),
                tfk.layers.Flatten(input_shape=(no_of_filters, 1, 1)),
                tfk.layers.Dense(units=no_of_classes, input_shape=(no_of_filters,), activation=tf.nn.sigmoid)
            ])
        else:
            print("CustomNetworkHandler does not handle a model type called: ", model_name)
            print("Please one of the following names: 'sandbox, 'CNN'")

        # self.model.summary()

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[tfk.metrics.Recall(), tfk.metrics.Precision()])

    def train(self, input_array, label_array):
        self.model.fit(input_array,
                       label_array,
                       batch_size=32,
                       epochs=3)
        return

    def save_model_as_file(self, filename):
        tfk.models.save_model(
            self.model,
            'SavedModels/' + filename)

        return


