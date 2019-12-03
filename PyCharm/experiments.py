# General modules
import copy
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras as tfk

# Personal modules
import masking
import visualization
from network import CustomNetworkWrapper as NetworkWrapper
from datasets import reuters
from datasets import mnist


def search_for_lottery_tickets(epochs, model_identifier, pruning_percentage=20, pruning_iterations=1):
    print("Developing full " + model_identifier + "...")
    base_model_wrapper = NetworkWrapper(model_identifier=model_identifier)

    print("Quantifying datapoints...")
    data_splits = mnist.quantify_datapoints()
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Training full network...")
    full_prediction_history =\
        base_model_wrapper.train_model_with_sklearn_metrics(datapoints=train_datapoints,
                                                            validation_datapoints=test_datapoints,
                                                            epochs=epochs,
                                                            batch_size=60,
                                                            verbosity=2)

    masked_prediction_histories = []
    for i in range(0, pruning_iterations):
        print("-"*15 + " Pruning Iteration " + str(i) + " " + "-"*15)
        # Read out the config for creation of the masked model
        base_model_config = base_model_wrapper.model.get_config()
        # Copy original weights by value
        base_weights = []
        base_biases = []
        for i in range(len(base_model_wrapper.model.weights)):
            if (i % 2) == 0:
                base_weights.append(tf.identity(base_model_wrapper.model.weights[i]))
            elif (i % 2) == 1:
                base_biases.append(tf.identity(base_model_wrapper.model.weights[i]))
            else:
                print("Separation weights and biases failed!")

        print("Developing the masked network...")
        masked_model = masking.mask_initial_model(trained_model=base_model_wrapper.model,
                                                  initial_weights=base_weights,
                                                  initial_biases=base_biases,
                                                  model_config=base_model_config,
                                                  pruning_percentage=20)

        lottery_ticket_wrapper = NetworkWrapper(model_identifier='GivenModel',
                                                given_model=masked_model)

        print("Training masked network...")
        masked_prediction_history =\
            lottery_ticket_wrapper.train_model_with_sklearn_metrics(datapoints=train_datapoints,
                                                                    validation_datapoints=test_datapoints,
                                                                    epochs=epochs,
                                                                    batch_size=60,
                                                                    verbosity=2)
        masked_prediction_histories.append(masked_prediction_history)

        # Set this masked network as base for the next iteration
        base_model_wrapper = lottery_ticket_wrapper

    return full_prediction_history, masked_prediction_histories


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



