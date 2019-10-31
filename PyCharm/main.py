from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras.saving.saved_model import load as tfk_load

# Personal modules
import masking
from network import CustomNetworkWrapper as NetworkWrapper
from custom_layers import MaskedDense
from datasets import reuters
# tf.debugging.set_log_device_placement(True)


def main():
    # For debugging
    # tf.config.experimental_run_functions_eagerly(True)

    # Hack to prevent a specific error with cudNN
    # https://github.com/tensorflow/tensorflow/issues/24828
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    reuters_model_wrapper = NetworkWrapper(no_of_features=0,
                                           model_identifier='GivenModel',
                                           given_model=tfk.models.load_model('SavedModels/test-trained'))

    # reuters_model.save_model_as_file('test-trained')

    data_splits = reuters.quantify_datapoints(target_no_of_features=30)
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    lottery_ticket = construct_lottery_ticket(trained_model=reuters_model_wrapper.model,
                                              init_model=tfk.models.load_model('SavedModels/test-init'))

    lottery_ticket_wrapper = NetworkWrapper(no_of_features=0,
                                            model_identifier='GivenModel',
                                            given_model=lottery_ticket)

    lottery_ticket_wrapper.train_model(datapoints=train_datapoints,
                                       epochs=3)

    reuters_model_wrapper.evaluate_model(test_datapoints)
    lottery_ticket_wrapper.evaluate_model(test_datapoints)

    return


def construct_lottery_ticket(trained_model, init_model):
    masks = masking.create_masks(trained_model)

    init_model.summary()
    model_config = init_model.get_config()

    masked_model = tfk.Sequential()
    for idx, layer in enumerate(init_model.layers):
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


def construct_model_handler_for_reuters():
    ((batched_train_words, batched_train_cats),
     (batched_test_words, batched_test_cats)) = reuters.quantify_datapoints(target_no_of_features=30)

    print("Developing network...")
    model_handler = NetworkWrapper(no_of_features=30,
                                   model_identifier='FeedForward')
    # Add a channel dimension for CNNs
    # batched_train_words = np.reshape(batched_train_words, np.shape(batched_train_words) + (1,))
    # batched_test_words = np.reshape(batched_test_words, np.shape(batched_test_words) + (1,))

    # model_handler.save_model_as_file('test-init')

    print("Training network...")
    model_handler.train_model(datapoints=(batched_train_words, batched_train_cats),
                              epochs=3)

    return model_handler


main()
