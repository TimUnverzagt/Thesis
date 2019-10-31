# General modules
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk

# Personal modules
import masking
from network import CustomNetworkWrapper as NetworkWrapper
from datasets import reuters


def test_creation_of_masked_network():
    reuters_model_wrapper = NetworkWrapper(no_of_features=0,
                                           model_identifier='GivenModel',
                                           given_model=tfk.models.load_model('SavedModels/test-trained'))

    # reuters_model.save_model_as_file('test-trained')

    data_splits = reuters.quantify_datapoints(target_no_of_features=30)
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    lottery_ticket = masking.mask_initial_model_ticket(trained_model=reuters_model_wrapper.model,
                                                       initial_model=tfk.models.load_model('SavedModels/test-init'))

    lottery_ticket_wrapper = NetworkWrapper(no_of_features=0,
                                            model_identifier='GivenModel',
                                            given_model=lottery_ticket)

    lottery_ticket_wrapper.train_model(datapoints=train_datapoints,
                                       epochs=3)

    reuters_model_wrapper.evaluate_model(test_datapoints)
    lottery_ticket_wrapper.evaluate_model(test_datapoints)


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

