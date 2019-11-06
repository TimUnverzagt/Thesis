# General modules
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk

# Personal modules
import masking
from network import CustomNetworkWrapper as NetworkWrapper
from datasets import reuters
from datasets import mnist


def test_basic_network_of_the_paper(epochs):
    print("Developing feedforward network on MNIST...")
    dense_model_wrapper = NetworkWrapper(model_identifier='MNIST-Lenet-FC')
    initial_model = dense_model_wrapper.model

    print("Quantifying MNIST datapoints...")
    data_splits = mnist.quantify_datapoints()
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Training full network...")
    full_history = dense_model_wrapper.train_model(datapoints=train_datapoints,
                                                   epochs=epochs,
                                                   batch_size=60)

    print("Developing a masked network with the initial weights...")
    masked_model = masking.mask_initial_model(trained_model=dense_model_wrapper.model,
                                              initial_model=initial_model,
                                              pruning_percentage=20)
    lottery_ticket_wrapper = NetworkWrapper(model_identifier='GivenModel',
                                            given_model=masked_model)

    print("Training masked network...")
    masked_history = lottery_ticket_wrapper.train_model(datapoints=train_datapoints,
                                                        epochs=epochs,
                                                        batch_size=60)

    print("Quickly evaluating both networks...")
    dense_model_wrapper.evaluate_model(test_datapoints)
    lottery_ticket_wrapper.evaluate_model(test_datapoints)

    return full_history, masked_history


def test_creation_of_masked_network(epochs):
    """
    reuters_model_wrapper = NetworkWrapper(no_of_features=0,
                                           model_identifier='GivenModel',
                                           given_model=tfk.models.load_model('SavedModels/test-trained'))
    """
    print("Developing feedforward network on reuters...")
    reuters_model_wrapper = NetworkWrapper(no_of_features=30,
                                           model_identifier='Reuters-FeedForward')
    initial_model = reuters_model_wrapper.model

    print("Quantifying reuters datapoints...")
    data_splits = reuters.quantify_datapoints(target_no_of_features=30)
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Training full network...")
    full_history = reuters_model_wrapper.train_model(datapoints=train_datapoints,
                                                     epochs=epochs)

    print("Developing a masked network with the initial weights...")
    masked_model = masking.mask_initial_model(trained_model=reuters_model_wrapper.model,
                                              initial_model=initial_model)
    lottery_ticket_wrapper = NetworkWrapper(no_of_features=0,
                                            model_identifier='GivenModel',
                                            given_model=masked_model)

    print("Training masked network...")
    masked_history = lottery_ticket_wrapper.train_model(datapoints=train_datapoints,
                                                        epochs=epochs)

    print("Quickly evaluating both networks...")
    reuters_model_wrapper.evaluate_model(test_datapoints)
    lottery_ticket_wrapper.evaluate_model(test_datapoints)

    return full_history, masked_history

