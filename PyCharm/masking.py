from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from custom_layers import MaskedDense
from custom_layers import MaskedConv2D


def save_trainable_values(model_wrapper):
    config = model_wrapper.model.get_config()
    n_o_trainable_layers_per_submodel = []
    for subconfig in config['layers']:
        n_o_trainable_layers_per_submodel.append(
            _find_amount_of_values_for_subconfig(subconfig)
        )
    save_values = []
    n_o_already_saved_values = 0
    for n_o_trainable_layers in n_o_trainable_layers_per_submodel:
        save_values_per_submodel = []
        for i in range(n_o_trainable_layers):
            save_values_per_submodel.append(
                (tf.identity(model_wrapper.model.weights[n_o_already_saved_values]))
            )
            n_o_already_saved_values += 1
        save_values.append(save_values_per_submodel)
    '''
    if _is_functional_config(config):
        print()
    elif _is_sequential_subconfig():
        print()
    elif _is_layer_subconfig():
        print()
    else:
        print("Failed to read out config while saving values")
    '''
    '''
    base_weights = []
    base_biases = []
    for j in range(len(model_wrapper.model.weights)):
        if (j % 2) == 0:
            base_weights.append(tf.identity(model_wrapper.model.weights[j]))
        elif (j % 2) == 1:
            base_biases.append(tf.identity(model_wrapper.model.weights[j]))
        else:
            print("Separation of weights and biases failed!")
    '''
    # return base_weights, base_biases
    return save_values


def _find_amount_of_values_for_subconfig(config):
    no_values = 0
    if _is_functional_config(config):
        for subconfig in config['layers']:
            no_values += _find_amount_of_values_for_subconfig(subconfig)

    elif _is_sequential_subconfig(config):
        layers = config['config']['layers']
        for layer_config in layers:
            no_values += int(_is_layer_with_weights(layer_config))
            no_values += int(_is_layer_with_bias(layer_config))
    elif _is_layer_subconfig(config):
        no_values += int(_is_layer_with_weights(config))
        no_values += int(_is_layer_with_bias(config))
    else:
        print("Failed to read out config while saving values")
    return no_values


def mask_model(trained_model, initial_weights, initial_biases, model_config, pruning_percentages,
               layer_wise=False, summarize=False):
    # TODO: Rewrite code to handle specific non-dense pruning percentages more flexibly
    if not (pruning_percentages['conv'] < 1.0e-3):
        print("A special pruning percentage for convolutional layers is currently only supported layer_wise.")
        print("Mask creation has been set to layer_wise!")
        layer_wise = True

    # masks = _create_masks_for_sequential_model(trained_model, pruning_percentages, layer_wise=layer_wise)
    masks = _create_mask_for_functional_model(trained_model, pruning_percentages, layer_wise=True)

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


def _create_masks_for_sequential_model(trained_model, pruning_percentages, layer_wise):
    # TODO: This is a mess
    # TODO: Layer-wise pruning should not change whether one can apply different percentages for different weights
    # TODO: At the moment the inability to differentiate weights globally is the bottleneck (which can be easily fixed)
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


def _create_mask_for_functional_model(trained_model, pruning_percentages, layer_wise):
    masked_submodels = ()
    main_config = trained_model.get_config()

    masks = []
    for subconfig in main_config['layers']:
        if _is_functional_config(subconfig):
            print("Detected subconfig in functional shape")
        elif _is_sequential_subconfig(subconfig):
            print("Detected subconfig in sequential shape")
        elif _is_layer_subconfig(subconfig):
            print("Detected subconfig in layer shape")
            if _is_maskable_layer(subconfig):
                masks.append(_create_mask_for_layer(subconfig, pruning_percentages, ))
    return 0


def _create_mask_for_layer(config, pruning_percentages, weights):
    return


def _magnitude_threshold_mask(values, threshold):
    print("Weight of the threshold for masking: ", np.round(threshold, 4))
    # Does this work as intended with numeric errors?
    return tf.cast(tf.map_fn(lambda x: abs(x) >= threshold, values, bool), tf.int32)


def _is_maskable_layer(config):
    maskable = False
    maskable = (config['class_name'] == 'Dense') | maskable
    maskable = (config['class_name'] == 'MaskedDense') | maskable
    maskable = (config['class_name'] == 'Conv2D') | maskable
    maskable = (config['class_name'] == 'MaskedConv2D') | maskable
    return maskable


def _is_layer_with_weights(config):
    has_weights = _is_maskable_layer(config)
    has_weights = (config['class_name'] == 'Embedding') | has_weights
    has_weights = (config['class_name'] == 'Conv1D') | has_weights
    return has_weights


def _is_layer_with_bias(config):
    has_bias = _is_layer_with_weights(config)
    # Remove layers with weights only
    has_bias = (not (config['class_name'] == 'Embedding')) & has_bias
    return has_bias


def _is_functional_config(config):
    # TODO: There might be more robust ways of identification
    return 'input_layers' in config


def _is_sequential_subconfig(subconfig):
    # TODO: There might be more robust ways of identification
    return 'Sequential' == subconfig['class_name']


def _is_layer_subconfig(subconfig):
    # TODO: There might be more robust ways of identification
    return 'layers' not in subconfig['config']
