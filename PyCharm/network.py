from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class CustomNetworkHandler:

    def __init__(self, target_doc_len):
        middle_size = np.round(np.sqrt(target_doc_len * 300 * 90))
        self.model = tfk.Sequential([
            tfk.layers.Input(shape=(target_doc_len, 300)),
            tfk.layers.Flatten(input_shape=(target_doc_len, 300)),
            # 300 = dimensionality of embedding
            tfk.layers.Dense(300 * target_doc_len, activation=tf.nn.relu),
            # middle layer is chosen so the downscaling factor is constant
            tfk.layers.Dense(middle_size, activation=tf.nn.relu),
            # 90 = number of categories
            tfk.layers.Dense(90, activation=tf.nn.sigmoid)
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[tfk.metrics.Recall(), 'accuracy'])

    def train(self, input_array, label_array):
        self.model.fit(input_array,
                       label_array,
                       batch_size=32,
                       epochs=5)
        return
