from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


def mask_network():
    init_model = tfk.models.load_model('SavedModels/test-init')
    init_model.summary()
    percentile = 0
    flattened_weights = None
    weights_are_initialized = False
    for layer in init_model.layers:
        if layer.weights:
            if not weights_are_initialized:
                flattened_weights = tf.reshape(layer.weights[0], [-1])
            else:
                flattened_weights = tf.concat(flattened_weights, tf.reshape(layer.weights[0], [-1]))
    percentile = np.percentile(flattened_weights.numpy(), 90)
    print(percentile)

    trained_model = tfk.models.load_model('SavedModels/test-trained')
    trained_model.summary()
    return
