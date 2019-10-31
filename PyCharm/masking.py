from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from custom_layers import MaskedDense


def mask_initial_model_ticket(trained_model, initial_model, pruning_percentage=50):
    masks = _create_masks(trained_model, pruning_percentage)

    initial_model.summary()
    model_config = initial_model.get_config()

    masked_model = tfk.Sequential()
    for idx, layer in enumerate(initial_model.layers):
        print(model_config['layers'][idx]['class_name'])
        if model_config['layers'][idx]['class_name'] == 'Dense':
            print("Replacing Dense-layer of the model with a custom MaskedDense-layer")
            if model_config['layers'][idx]['config']['activation'] == 'relu':
                print("Recognized an relu-activation.")
                old_activation = tf.nn.relu
            elif model_config['layers'][idx]['config']['activation'] == 'sigmoid':
                print("Recognized an sigmoid-activation.")
                old_activation = tf.nn.sigmoid
            else:
                # TODO: Throw real exception
                print('The activation of the given model is not recognized.')
                print('No activation was chosen. This will likely result in a critical error!')

            replacement_layer = MaskedDense(units=layer.output_shape[1],
                                            activation=old_activation,
                                            kernel=layer.kernel,
                                            mask=masks[idx],
                                            bias=layer.bias
                                            )
            masked_model.add(replacement_layer)
        else:
            masked_model.add(layer)

    masked_model.build()
    masked_model.summary()
    return masked_model


def _create_masks(trained_model, pruning_percentage):
    flattened_weights = None
    weights_are_initialized = False
    for layer in trained_model.layers:
        if layer.weights:
            if not weights_are_initialized:
                flattened_weights = tf.reshape(layer.weights[0], [-1])
            else:
                flattened_weights = tf.concat(flattened_weights, tf.reshape(layer.weights[0], [-1]))

    percentile = np.percentile(np.abs(flattened_weights.numpy()), pruning_percentage)
    print(percentile)

    # List of tuples containing the idx and mask for each layer with trainable weights
    masks = {}
    for idx, layer in enumerate(trained_model.layers):
        if layer.weights:
            # Only mask the weight-kernel (weights[0]) not the biases (weights[1])
            mask = _magnitude_threshold_mask(layer.weights[0], percentile)
            masks[idx] = mask
    return masks


def _magnitude_threshold_mask(values, threshold):
    return tf.cast(tf.map_fn(lambda x: x >= threshold, values, bool), tf.int32)
