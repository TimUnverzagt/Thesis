from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


def create_masks(trained_model):
    percentile = 0
    flattened_weights = None
    weights_are_initialized = False
    for layer in trained_model.layers:
        if layer.weights:
            if not weights_are_initialized:
                flattened_weights = tf.reshape(layer.weights[0], [-1])
            else:
                flattened_weights = tf.concat(flattened_weights, tf.reshape(layer.weights[0], [-1]))

    percentile = np.percentile(np.abs(flattened_weights.numpy()), 50)
    print(percentile)

    # List of tuples containing the idx and mask for each layer with trainable weights
    masks = {}
    for idx, layer in enumerate(trained_model.layers):
        if layer.weights:
            # Only mask the weight-kernel (weights[0]) not the biases (weights[1])
            mask = _threshold_mask(layer.weights[0], percentile)
            masks[idx] = mask
    return masks


def _threshold_mask(values, threshold):
    return tf.cast(tf.map_fn(lambda x: x >= threshold, values, bool), tf.int32)
