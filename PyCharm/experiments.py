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
    print("Quantifying datapoints...")
    data_splits = mnist.quantify_datapoints()
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Developing full " + model_identifier + "...")
    base_model_wrapper = NetworkWrapper(model_identifier=model_identifier)
    # Read out the config for creation of the masked model
    base_model_config = base_model_wrapper.model.get_config()
    # Copy original weights by value
    base_weights, base_biases = custom_weight_copy(base_model_wrapper)
    print("Training full network...")
    full_prediction_history = inspect_metrics_while_training(model_wrapper=base_model_wrapper,
                                                             training_data=train_datapoints,
                                                             validation_data=test_datapoints,
                                                             epochs=epochs,
                                                             batch_size=60,
                                                             verbosity=2)
    masked_prediction_histories = []

    for i in range(0, pruning_iterations):
        print("-"*15 + " Pruning Iteration " + str(i) + " " + "-"*15)
        print("Developing the masked network...")
        masked_model = masking.mask_initial_model(trained_model=base_model_wrapper.model,
                                                  initial_weights=base_weights,
                                                  initial_biases=base_biases,
                                                  model_config=base_model_config,
                                                  pruning_percentage=20)
        lottery_ticket_wrapper = NetworkWrapper(model_identifier='GivenModel',
                                                given_model=masked_model)

        # Set this masked network as base for the next iteration
        base_model_wrapper = lottery_ticket_wrapper
        # Read out the config for creation of the masked model
        base_model_config = base_model_wrapper.model.get_config()
        # Copy original weights by value
        base_weights, base_biases = custom_weight_copy(base_model_wrapper)

        print("Training masked network...")
        masked_prediction_history = inspect_metrics_while_training(model_wrapper=lottery_ticket_wrapper,
                                                                   training_data=train_datapoints,
                                                                   validation_data=test_datapoints,
                                                                   epochs=epochs,
                                                                   batch_size=60,
                                                                   verbosity=2)
        masked_prediction_histories.append(masked_prediction_history)
    return full_prediction_history, masked_prediction_histories


def search_early_ticket(epochs, model_identifier, reset_epochs=10, pruning_percentage=20, pruning_iterations=1):
    print("Quantifying datapoints...")
    data_splits = mnist.quantify_datapoints()
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Developing full " + model_identifier + "...")
    base_model_wrapper = NetworkWrapper(model_identifier=model_identifier)
    intermediate_weights = []
    intermediate_weights.append(custom_weight_copy(base_model_wrapper))

    for i in range(1, reset_epochs+1):
        # Read out the config for creation of the masked model
        base_model_config = base_model_wrapper.model.get_config()
        # Copy original weights by value
        base_weights = custom_weight_copy(base_model_wrapper)
        print("Training full network...")

        full_prediction_history = inspect_metrics_while_training(model_wrapper=base_model_wrapper,
                                                                 training_data=train_datapoints,
                                                                 validation_data=test_datapoints,
                                                                 epochs=1,
                                                                 batch_size=60,
                                                                 verbosity=2)

    return


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


def inspect_metrics_while_training(model_wrapper, training_data, validation_data, epochs, batch_size,
                                   verbosity=2, scheme_for_averaging="micro"):

    accuracy_over_epochs = []
    precision_over_epochs = []
    recall_over_epochs = []
    confusion_matrices = []
    sparse_true_labels = sparsify_predictions(validation_data[1])
    for i in range(epochs):
        print(12 * "-" + "Epoch " + str(i) + 12 * "-")
        model_wrapper.train_model(training_data,
                                  batch_size=batch_size,
                                  epochs=1,
                                  verbosity=verbosity)

        sparse_predictions = sparsify_predictions(model_wrapper.model.predict(validation_data))
        accuracy_over_epochs.append(sklearn.metrics.accuracy_score(sparse_true_labels, sparse_predictions))
        precision_over_epochs.append(sklearn.metrics.precision_score(sparse_true_labels,
                                                                     sparse_predictions,
                                                                     average=scheme_for_averaging))
        recall_over_epochs.append(sklearn.metrics.recall_score(sparse_true_labels,
                                                               sparse_predictions,
                                                               average=scheme_for_averaging))
        confusion_matrices.append(sklearn.metrics.confusion_matrix(sparse_true_labels, sparse_predictions))
    print(31 * "-")

    return {'accuracy': accuracy_over_epochs,
            'precision': precision_over_epochs,
            'recall': recall_over_epochs,
            'confusion_matrices': confusion_matrices}


def custom_weight_copy(model_wrapper):
    base_weights = []
    base_biases = []
    for j in range(len(model_wrapper.model.weights)):
        if (j % 2) == 0:
            base_weights.append(tf.identity(model_wrapper.model.weights[j]))
        elif (j % 2) == 1:
            base_biases.append(tf.identity(model_wrapper.model.weights[j]))
        else:
            print("Separation weights and biases failed!")
        return (base_weights, base_biases)


def sparsify_predictions(one_hot_predictions):
    return np.argmax(one_hot_predictions, axis=1)


