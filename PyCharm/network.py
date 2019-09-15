from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class CustomNetworkHandler:

    def __init__(self, doc_len):
        middle_size = np.round(np.sqrt(doc_len * 300 * 90))
        self.model = tfk.Sequential([
            tfk.layers.Input(shape=(doc_len, 300)),
            tfk.layers.Flatten(input_shape=(doc_len, 300)),
            # 300 = dimensionality of embedding
            tfk.layers.Dense(300 * doc_len, activation=tf.nn.relu),
            # middle layer is chosen so the downscaling factor is constant
            tfk.layers.Dense(middle_size, activation=tf.nn.relu),
            # 90 = number of categories
            tfk.layers.Dense(90, activation=tf.nn.sigmoid)
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self, input_array, annotation_array):
        self.model.fit(input_array,
                       annotation_array,
                       batch_size=32,
                       epochs=10)
        return
