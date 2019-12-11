from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from custom_layers import MaskedDense
from custom_layers import MaskedConv2D


def mask_model(trained_model, initial_weights, initial_biases, model_config, pruning_percentages,
               layer_wise=False, summarize=False):
    if not ((pruning_percentages['conv'] < 1.0e-3) | (pruning_percentages['conv'] < 1.0e-3)):
        print("Different pruning percentages are currently only supported layer_wise.")
        print("Mask creation has been set to layer_wise!")
        masks = _create_masks(trained_model, pruning_percentages, layer_wise=True)
    else:
        masks = _create_masks(trained_model, pruning_percentages, layer_wise=layer_wise)

    masked_model = tfk.Sequential()
    weight_idx = 0
    for layer in iter(model_config['layers']):
        config = layer['config']

        if layer['class_name'] == 'Flatten':
            if 'batch_input_shape' in config:
                masked_model.add(tfk.layers.Flatten(batch_input_shape=config['batch_input_shape']))
            else:
                masked_model.add(tfk.layers.Flatten())

        elif layer['class_name'] == 'MaxPooling2D':
            masked_model.add(tfk.layers.MaxPool2D())

        elif (layer['class_name'] == 'Conv2D') | (layer['class_name'] == 'MaskedConv2D'):
            print("Replacing Conv2D-layer of the model with a custom MaskedConv2D-layer")
            if 'batch_input_shape' in config:
                masked_model.add(MaskedConv2D(
                    filters=config['filters'],
                    kernel_size=config['kernel_size'],
                    padding=config['padding'],
                    activation=config['activation'],
                    batch_input_shape=config['batch_input_shape'],
                    initialization_weights=initial_weights[weight_idx],
                    mask=masks[weight_idx],
                    initialization_bias=initial_biases[weight_idx])
                )
            else:

                masked_model.add(MaskedConv2D(
                    filters=config['filters'],
                    kernel_size=config['kernel_size'],
                    padding=config['padding'],
                    activation=config['activation'],
                    initialization_weights=initial_weights[weight_idx],
                    mask=masks[weight_idx],
                    initialization_bias=initial_biases[weight_idx])
                )
            weight_idx += 1

        elif (layer['class_name'] == 'Dense') | (layer['class_name'] == 'MaskedDense'):
            print("Replacing Dense-like-layer of the model with a custom MaskedDense-layer")
            masked_model.add(MaskedDense(units=config['units'],
                                         activation=config['activation'],
                                         initialization_weights=initial_weights[weight_idx],
                                         mask=masks[weight_idx],
                                         initialization_bias=initial_biases[weight_idx]
                                         ))
            weight_idx += 1
        else:
            print("Layer " + layer['class_name'] + " not recognized. Fatal Error imminent!")

    masked_model.build()
    if summarize:
        masked_model.summary()

    return masked_model


def _create_masks(trained_model, pruning_percentages, layer_wise):
    if layer_wise:
        masks = []
        for layer in trained_model.layers:
            if hasattr(layer, 'weights') & bool(layer.weights):
                if hasattr(layer, 'mask'):
                    weights = tf.multiply(layer.weights[0], layer.mask)
                else:
                    weights = layer.weights[0]
                # Hacky way to check whether we handle a dense or conv layer
                if hasattr(layer, 'filters'):
                    quantile = np.percentile(np.abs(weights.numpy()), pruning_percentages['conv'])
                else:
                    quantile = np.percentile(np.abs(weights.numpy()), pruning_percentages['dense'])
                masks.append(_magnitude_threshold_mask(weights, quantile))

    else:
        flattened_weights = None
        weights_are_initialized = False
        for layer in trained_model.layers:
            if hasattr(layer, 'weights') & bool(layer.weights):
                if not weights_are_initialized:
                    if hasattr(layer, 'mask'):
                        flattened_weights = tf.reshape(tf.multiply(layer.weights[0], layer.mask), [-1])
                    else:
                        flattened_weights = tf.reshape(layer.weights[0], [-1])
                else:
                    if hasattr(layer, 'mask'):
                        flattened_weights = \
                            tf.concat(flattened_weights, tf.reshape(tf.multiply(layer.weights[0], layer.mask), [-1]))
                    else:
                        flattened_weights = tf.concat(flattened_weights, tf.reshape(layer.weights[0], [-1]))
        quantile = np.percentile(np.abs(flattened_weights.numpy()), pruning_percentages['dense'])
        # List of tuples containing the idx and mask for each layer with trainable weights
        masks = []
        for layer in iter(trained_model.layers):
            if layer.weights:
                # Only mask the weight-kernel (weights[0]) not the biases (weights[1])
                mask = _magnitude_threshold_mask(layer.weights[0], quantile)
                masks.append(mask)

    return masks


def _magnitude_threshold_mask(values, threshold):
    print("Weight of the threshold for masking: ", np.round(threshold, 4))
    # Does this work as intended with numeric errors?
    return tf.cast(tf.map_fn(lambda x: abs(x) >= threshold, values, bool), tf.int32)
