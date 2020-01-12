from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import time
from collections import Iterable

# Personal modules
from custom_layers import MaskedDense
from custom_layers import MaskedConv1D
from custom_layers import MaskedConv2D
from custom_layers import MaskedEmbedding


def extract_trainable_values(model):
    config = model.get_config()
    weights = model.weights

    # Go over the config recursively once to understand how the value tensors are spread
    no_of_tensors = _find_no_of_tensors_rec(config)

    # Go over all tensors to recursevly associate them to the structure
    structured_tensors = _structurize_tensors_rec(
        config,
        tensor_distribution=no_of_tensors,
        tensors=weights)

    return structured_tensors


def _structurize_tensors_rec(config, tensor_distribution, tensors):
    structured_tensors = []
    if _is_functional_config(config):
        for idx, wrapped_config in enumerate(config['layers']):
            subconfig = wrapped_config['config']
            subconfig['class_name'] = wrapped_config['class_name']
            if isinstance(tensor_distribution[idx], Iterable):
                no_needed_tensors = sum(_flatten(tensor_distribution[idx]))
            else:
                no_needed_tensors = tensor_distribution[idx]

            no_used_tensors = sum(_flatten(tensor_distribution[0:idx]))
            subtensors = tensors[no_used_tensors: (no_used_tensors + no_needed_tensors)]
            structured_tensors.append(
                _structurize_tensors_rec(
                    subconfig,
                    tensor_distribution[idx],
                    subtensors)
            )

    elif _is_sequential_config(config):
        layers = config['layers']
        no_used_tensors = 0
        for idx, wrapped_config in enumerate(layers):
            layer_config = wrapped_config['config']
            layer_config['class_name'] = wrapped_config['class_name']
            wb_dict = {}
            if _is_layer_with_weights(layer_config):
                wb_dict['weights'] = tf.identity(tensors[no_used_tensors])
                no_used_tensors += 1
                if _is_layer_with_biases(layer_config):
                    wb_dict['biases'] = tf.identity(tensors[no_used_tensors])
                    no_used_tensors += 1
            structured_tensors.append(wb_dict)

    elif _is_layer_config(config):
        eb_dict = {}
        if _is_layer_with_weights(config):
            eb_dict['weights'] = tf.identity(tensors[0])
            if _is_layer_with_biases(config):
                eb_dict['biases'] = tf.identity(tensors[1])
        structured_tensors = eb_dict
    return structured_tensors


def _find_no_of_tensors_rec(config):
    no_tensor = None
    if _is_functional_config(config):
        no_tensor = []
        for wrapped_config in config['layers']:
            subconfig = wrapped_config['config']
            subconfig['class_name'] = wrapped_config['class_name']
            no_tensor.append(_find_no_of_tensors_rec(subconfig))

    elif _is_sequential_config(config):
        no_tensor = 0
        layers = config['layers']
        for wrapped_config in layers:
            layer_config = wrapped_config['config']
            layer_config['class_name'] = wrapped_config['class_name']
            no_tensor += int(_is_layer_with_weights(layer_config))
            no_tensor += int(_is_layer_with_biases(layer_config))

    elif _is_layer_config(config):
        no_tensor = int(_is_layer_with_weights(config)) + int(_is_layer_with_biases(config))
    return no_tensor


def mask_model(trained_model, initial_values, pruning_percentages, summarize=True):
    config = trained_model.get_config()
    masks = _create_mask_for_functional_model(config, initial_values,
                                              pruning_percentages)
    if _is_sequential_config(config):
        input_layer = tfk.layers.Input(shape=trained_model.input_shape[1:])
        masked_model = _build_masked_model(
            config, initial_values, masks,
            has_sequential_base=True,
            sequential_input=input_layer
        )
    else:
        masked_model = _build_masked_model(config, initial_values, masks)

    masked_model.compile(
        optimizer=trained_model.optimizer,
        loss=trained_model.loss
    )
    if summarize:
        masked_model.summary()
    return masked_model


def _build_masked_model(config, initial_values, masks, has_sequential_base=False, sequential_input=None):
    masked_model = []
    if _is_functional_config(config):
        print("===Begin recreation of a functional model===")
        submodels = []
        submodel_idxs = {}
        for idx, wrapped_config in enumerate(config['layers']):
            # Identify the layers that act as inputs for the submodel
            input_layer = []
            if wrapped_config['inbound_nodes']:
                for inbound_node in wrapped_config['inbound_nodes'][0]:
                    layer_names = inbound_node[0]
                    input_layer_idxs = submodel_idxs[layer_names]
                    input_layer.append(submodels[input_layer_idxs])
            # Remember where to submodul was placed by name so it can later be handed over as input
            submodel_idxs[wrapped_config['name']] = idx
            subconfig = wrapped_config['config']
            subconfig['class_name'] = wrapped_config['class_name']
            masked_submodel = _build_masked_model(
                config=subconfig,
                initial_values=initial_values[idx],
                masks=masks[idx]
            )
            if input_layer:
                if len(input_layer) == 1:
                    input_layer = input_layer[0]
                submodels.append(masked_submodel(input_layer))
            else:
                submodels.append(masked_submodel)

        # Collect everything into a functional model by defining global inputs and outputs
        input_name = config['input_layers'][0][0]
        input_layer_idxs = submodel_idxs[input_name]
        input_layer = submodels[input_layer_idxs]
        output_name = config['output_layers'][0][0]
        output_layer_idx = submodel_idxs[output_name]
        output_layer = submodels[output_layer_idx]

        masked_model = tfk.Model(inputs=input_layer,
                                 outputs=output_layer)
        print("===End recreation of a functional model===")

    elif _is_sequential_config(config):
        print("---Begin recreation sequential model---")
        layers = config['layers']
        masked_model = tfk.Sequential()
        if has_sequential_base:
            masked_model.add(sequential_input)
        no_weights_used = 0
        for idx, wrapped_config in enumerate(layers):
            print("Recreate single sequential layer")
            layer_config = wrapped_config['config']
            layer_config['class_name'] = wrapped_config['class_name']
            if _is_maskable_layer(wrapped_config):
                masked_model.add(_produce_masked_layer(
                    layer_config=layer_config,
                    wb_dict=initial_values[idx],
                    mask=masks[idx]
                ))
            else:
                masked_model.add(_reproduce_layer(
                    layer_config=layer_config
                ))
        print("---End recreation sequential model---")

    elif _is_layer_config(config):
        print("Recreate single functional layer")
        if _is_maskable_layer(config):
            masked_model = _produce_masked_layer(
                layer_config=config,
                wb_dict=initial_values,
                mask=masks[0]
            )
        else:
            masked_model = _reproduce_layer(layer_config=config)
    return masked_model


def _reproduce_layer(layer_config):
    print("Reproducing Layer")
    # Extract singular Inputs to ensure the correct size of the inputs

    if _is_input_config(layer_config):
        reproduced_layer = tfk.layers.Input(
            shape=layer_config['batch_input_shape'][1:],
            batch_size=layer_config['batch_input_shape'][0]
        )

    elif _is_avg_pool_1d_config(layer_config):
        reproduced_layer = tfk.layers.AvgPool1D(
            pool_size=layer_config['pool_size']
        )

    elif _is_global_avg_pool_1d_config(layer_config):
        reproduced_layer = tfk.layers.GlobalAveragePooling1D()

    elif _is_dropout_config(layer_config):
        reproduced_layer = tfk.layers.Dropout(
            rate=layer_config['rate']
        )

    elif _is_concatenate_config(layer_config):
        reproduced_layer = tfk.layers.Concatenate()

    elif _is_max_pool_2d_config(layer_config):
        reproduced_layer = tfk.layers.MaxPool2D()

    elif _is_flatten_config(layer_config):
        reproduced_layer = tfk.layers.Flatten()

    else:
        print("### None of the following layers fit the config supplied: ###")
        print("### Input, AvgPool1D, GlobalAveragePooling1D, Dropout, .. ###")
        print("### Concatenate, MaxPool2D, Flatten ###")

    return reproduced_layer


def _produce_masked_layer(layer_config, wb_dict, mask):
    # TODO: Swap to masked variants xD
    if _is_conv_1d_config(layer_config):
        print("Replacing Conv1D-layer of the model with a custom MaskedConv1D-layer")
        reproduced_layer = MaskedConv1D(
            filters=layer_config['filters'],
            kernel_size=layer_config['kernel_size'],
            padding=layer_config['padding'],
            activation=layer_config['activation'],
            initialization_weights=wb_dict['weights'],
            initialization_bias=wb_dict['biases'],
            mask=mask
        )

    elif _is_conv_2d_config(layer_config):
        print("Replacing Conv2D-layer of the model with a custom MaskedConv2D-layer")
        reproduced_layer = MaskedConv2D(
            filters=layer_config['filters'],
            kernel_size=layer_config['kernel_size'],
            padding=layer_config['padding'],
            activation=layer_config['activation'],
            initialization_weights=wb_dict['weights'],
            initialization_bias=wb_dict['biases'],
            mask=mask
        )

    elif _is_dense_config(layer_config):
        print("Replacing Dense-layer of the model with a custom MaskedDense-layer")
        reproduced_layer = MaskedDense(
            units=layer_config['units'],
            activation=layer_config['activation'],
            initialization_weights=wb_dict['weights'],
            initialization_biases=wb_dict['biases'],
            mask=mask
        )

    elif _is_embedding_config(layer_config):
        print("MaskedEmbedding is not yet implemented")
        print("Embedding will simply be replaced")
        reproduced_layer = MaskedEmbedding(
            input_dim=layer_config['input_dim'],
            input_length=layer_config['input_length'],
            output_dim=layer_config['output_dim'],
            embeddings_initializer=layer_config['embeddings_initializer'],
            embeddings_regularizer=layer_config['embeddings_regularizer'],
            embeddings_constraint=layer_config['embeddings_constraint'],
            activity_regularizer=layer_config['activity_regularizer'],
            initialization_weights=wb_dict['weights'],
            mask=mask
        )

    else:
        print("### Although the config should belong to a maskable layer no specific one could be found. ###")
        print("### None of the following layers fit the config supplied: ###")
        print("### (Masked)Conv1D, (Masked)Conv2D, (Masked)Dense, .. ###")
        print("### (Masked)Embedding ###")
    return reproduced_layer


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


def _create_mask_for_functional_model(config, values, pruning_percentages):
    masks = []
    if _is_functional_config(config):
        print("=====Begin masking of a functional model=====")
        for idx, wrapped_config in enumerate(config['layers']):
            subconfig = wrapped_config['config']
            subconfig['class_name'] = wrapped_config['class_name']
            masks.append(_create_mask_for_functional_model(
                config=subconfig,
                values=values[idx],
                pruning_percentages=pruning_percentages)
            )
        print("=====End masking of a functional model=====")
    elif _is_sequential_config(config):
        print("-----Begin masking of a sequential model-----")
        for idx, wrapped_config in enumerate(config['layers']):
            layer_config = wrapped_config['config']
            layer_config['class_name'] = wrapped_config['class_name']
            if _is_maskable_layer(layer_config):
                print("Masking a sequential " + layer_config['class_name'] + "layer")
                # TODO: Extend to multiple pruning percentages
                tensor = values[idx]['weights']
                if _is_dense_config(layer_config) | _is_embedding_config(layer_config):
                    quantile = np.percentile(np.abs(tensor.numpy()), pruning_percentages['dense'])
                elif _is_conv_1d_config(layer_config) | _is_conv_2d_config(layer_config):
                    quantile = np.percentile(np.abs(tensor.numpy()), pruning_percentages['conv'])
                else:
                    print("### No pruning percentage could be assigned to maskable layer. ###")
                    print("### Critical error is imminent! ###")
                masks.append(_magnitude_threshold_mask(
                    tensor=tensor,
                    threshold=quantile
                ))
            else:
                masks.append(None)

        print("-----End masking of a sequential model-----")

    elif _is_layer_config(config):
        if _is_maskable_layer(config):
            print("Masking a functional " + config['class_name'] + "-layer")
            # TODO: Extend to multiple pruning percentages
            tensor = values['weights']
            quantile = np.percentile(np.abs(tensor.numpy()), pruning_percentages['conv'])
            masks.append(_magnitude_threshold_mask(
                tensor=tensor,
                threshold=quantile
            ))
        else:
            masks.append(None)
    return masks


def _magnitude_threshold_mask(tensor, threshold, verbosity=0):
    if verbosity > 1:
        print("Shape of input: " + str(tensor.shape))
        print("Weight of the threshold for masking: " + str(np.round(threshold, 4)))
    prev_time = time.time()
    mask = tensor.numpy()
    with np.nditer(mask, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = int(abs(x) >= threshold)
    # mask = tf.cast(tf.map_fn(lambda x: abs(x) >= threshold, tensor, bool), tf.int32)
    if verbosity > 1:
        print("Shape of mask: " + str(mask.shape))
        print("Time used: " + str(np.round((time.time() - prev_time), 4)))
        print("")
    return mask


def _is_maskable_layer(config):
    maskable = False
    maskable = _is_conv_1d_config(config) | maskable
    maskable = _is_conv_2d_config(config) | maskable
    maskable = _is_dense_config(config) | maskable
    maskable = _is_embedding_config(config) | maskable
    return maskable


def _is_layer_with_weights(config):
    has_weights = _is_maskable_layer(config)
    # If the code gets extended there might be discrepancies between
    # maskable layers and layers with weights.
    # These should be represented here.
    return has_weights


def _is_layer_with_biases(config):
    has_bias = _is_layer_with_weights(config)
    # Remove layers with weights only
    has_bias = (not _is_embedding_config(config)) & has_bias
    return has_bias


def _is_functional_config(config):
    return 'input_layers' in config


def _is_sequential_config(config):
    is_sequential = ('layers' in config) & (not _is_functional_config(config))
    if is_sequential:
        if 'inbound_nodes' in config:
            print("### The sequential config supplied is wrapped from a functional model. ###.")
            print("### For compatibility reasons only unwrapped sequential configs are expected. ###")
            print("### The wrapped config contains the unwrapped config under the key 'config'. ###")
            is_sequential = False
    return is_sequential


def _is_layer_config(config):
    # TODO: There might be more robust ways of identification
    is_layer = not ('layers' in config)
    if is_layer:
        if 'inbound_nodes' in config:
            print("### The layer config supplied is wrapped from a functional model. ###.")
            print("### For compatibility reasons only unwrapped layer configs are expected. ###")
            print("### The wrapped config contains the unwrapped config under the key 'config'. ###")
            is_layer = False
    return is_layer


def _is_input_config(layer_config):
    is_input = layer_config['class_name'] == 'InputLayer'
    is_input = (layer_config['class_name'] == 'Input') | is_input
    return is_input


def _is_avg_pool_1d_config(layer_config):
    is_avg_pool_1d = layer_config['class_name'] == 'AveragePooling1D'
    return is_avg_pool_1d


def _is_dropout_config(layer_config):
    is_dropout = layer_config['class_name'] == 'Dropout'
    return is_dropout


def _is_global_avg_pool_1d_config(layer_config):
    is_global_avg_pool_1d = layer_config['class_name'] == 'GlobalAveragePooling1D'
    return is_global_avg_pool_1d


def _is_concatenate_config(layer_config):
    is_concatenate = layer_config['class_name'] == 'Concatenate'
    return is_concatenate


def _is_max_pool_2d_config(layer_config):
    is_max_pool_2d = layer_config['class_name'] == 'MaxPooling2D'
    return is_max_pool_2d


def _is_flatten_config(layer_config):
    is_flatten = layer_config['class_name'] == 'Flatten'
    return is_flatten


def _is_conv_1d_config(layer_config):
    is_conv_1d = layer_config['class_name'] == 'Conv1D'
    is_conv_1d = (layer_config['class_name'] == 'MaskedConv1D') | is_conv_1d
    return is_conv_1d


def _is_conv_2d_config(layer_config):
    is_conv_2d = layer_config['class_name'] == 'Conv2D'
    is_conv_2d = (layer_config['class_name'] == 'MaskedConv2D') | is_conv_2d
    return is_conv_2d


def _is_dense_config(layer_config):
    is_dense = layer_config['class_name'] == 'Dense'
    is_dense = (layer_config['class_name'] == 'MaskedDense') | is_dense
    return is_dense


def _is_embedding_config(layer_config):
    is_embedding = layer_config['class_name'] == 'Embedding'
    is_embedding = (layer_config['class_name'] == 'MaskedEmbedding') | is_embedding
    return is_embedding


def _flatten(collection):
    for element in collection:
        if isinstance(element, Iterable):
            for x in _flatten(element):
                yield x
        else:
            yield element
