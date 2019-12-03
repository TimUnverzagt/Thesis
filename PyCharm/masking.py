from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from custom_layers import MaskedDense


def mask_initial_model(trained_model, initial_weights, initial_biases, model_config, pruning_percentage):
    masks = _create_masks(trained_model, pruning_percentage)

    masked_model = tfk.Sequential()
    weight_idx = 0
    for layer in iter(model_config['layers']):
        if layer['class_name'] == 'Flatten':
            replacement_layer = tfk.layers.Flatten(batch_input_shape=layer['config']['batch_input_shape'])
            masked_model.add(replacement_layer)

        elif (layer['class_name'] == 'Dense') | (layer['class_name'] == 'MaskedDense'):
            print("Replacing Dense-layer of the model with a custom MaskedDense-layer")
            if layer['config']['activation'] == 'relu':
                # print("Recognized an relu-activation.")
                old_activation = tf.nn.relu
            elif layer['config']['activation'] == 'sigmoid':
                # print("Recognized an sigmoid-activation.")
                old_activation = tf.nn.sigmoid
            elif layer['config']['activation'] == 'softmax':
                # print("Recognized an softmax-activation.")
                old_activation = tf.nn.softmax
            elif layer['config']['activation'] == 'linear':
                # print("Recognized an linear activation.")
                old_activation = None
            else:
                # TODO: Throw real exception
                print("The activation of the given model is not recognized.")
                print("No activation was chosen. This will likely result in a critical error!")

            replacement_layer = MaskedDense(units=layer['config']['units'],
                                            activation=old_activation,
                                            initialization_weights=initial_weights[weight_idx],
                                            mask=masks[weight_idx],
                                            initialization_bias=initial_biases[weight_idx]
                                            )
            masked_model.add(replacement_layer)
            weight_idx += 1
        else:
            print("Layer not recognized. Fatal Error imminent!")

    masked_model.build()
    # masked_model.summary()
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

    if pruning_percentage == 0:
        quantile = 0
    else:
        quantile = np.percentile(np.abs(flattened_weights.numpy()), pruning_percentage)
    print("Weight of the threshold for masking: ", np.round(quantile, 4))

    # List of tuples containing the idx and mask for each layer with trainable weights
    masks = []
    for layer in iter(trained_model.layers):
        if layer.weights:
            # Only mask the weight-kernel (weights[0]) not the biases (weights[1])
            mask = _magnitude_threshold_mask(layer.weights[0], quantile)
            masks.append(mask)
    return masks


def _magnitude_threshold_mask(values, threshold):
    # Does this work as intended with numeric errors?
    return tf.cast(tf.map_fn(lambda x: abs(x) >= threshold, values, bool), tf.int32)
