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
from datasets import newsgroups
from datasets import reuters
from datasets import mnist
from datasets import cifar10


def test_cnn_for_nlp(epochs, verbosity=1):
    print("Quantifying datapoints...")
    data_splits = newsgroups.quantify_datapoints(target_doc_len=200)
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    model_wrapper = NetworkWrapper(model_identifier='Newsgroups-End2End-CNN', summarize=True)

    full_prediction_history = inspect_metrics_while_training(model_wrapper=model_wrapper,
                                                             training_data=train_datapoints,
                                                             validation_data=test_datapoints,
                                                             epochs=epochs,
                                                             batch_size=60,
                                                             verbosity=verbosity)
    return


def search_lottery_tickets(epochs, model_identifier, pruning_percentages={'dense': 20, 'conv': 15},
                           pruning_iterations=1, verbosity=2):
    print("Quantifying datapoints...")
    # Identify dataset
    if 'MNIST' in model_identifier:
        data_splits = mnist.quantify_datapoints()
        one_hot_labeled = True
    elif 'CIFAR10' in model_identifier:
        data_splits = cifar10.quantify_datapoints()
        one_hot_labeled = True
    elif 'Newsgroups' in model_identifier:
        data_splits = newsgroups.quantify_datapoints(target_doc_len=200)
        one_hot_labeled = False
    else:
        print("Dataset was not recognized from the model_identifier.")
        print("A critical error is imminent!")

    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Developing full " + model_identifier + "...")
    base_model_wrapper = NetworkWrapper(model_identifier=model_identifier, summarize=True)
    # Read out the config for creation of the masked model
    base_model_config = base_model_wrapper.model.get_config()
    # Copy original weights by value
    base_values = masking.extract_trainable_values(base_model_wrapper.model)
    print("Training full network...")
    full_prediction_history = inspect_metrics_while_training(model_wrapper=base_model_wrapper,
                                                             training_data=train_datapoints,
                                                             validation_data=test_datapoints,
                                                             epochs=epochs,
                                                             batch_size=60,
                                                             verbosity=verbosity,
                                                             one_hot_labels=one_hot_labeled)
    masked_prediction_histories = []

    for i in range(1, pruning_iterations+1):
        iter_pruning_percentages = _iterate_pruning_percentages(percentages=pruning_percentages, iteration=i)
        _flag_pruning_iteration(i, iter_pruning_percentages)
        print("Developing the masked network...")
        masked_model = masking.mask_model(trained_model=base_model_wrapper.model,
                                          initial_values=base_values,
                                          pruning_percentages=iter_pruning_percentages)
        lottery_ticket_wrapper = NetworkWrapper(model_identifier='GivenModel',
                                                given_model=masked_model)

        # Set this masked network as base for the next iteration
        base_model_wrapper = lottery_ticket_wrapper
        # Read out the config for creation of the masked model
        base_model_config = base_model_wrapper.model.get_config()
        # Copy original weights by value
        base_values = masking.extract_trainable_values(base_model_wrapper.model)

        print("Training masked network...")
        masked_prediction_history = inspect_metrics_while_training(model_wrapper=lottery_ticket_wrapper,
                                                                   training_data=train_datapoints,
                                                                   validation_data=test_datapoints,
                                                                   epochs=epochs,
                                                                   batch_size=60,
                                                                   verbosity=verbosity,
                                                                   one_hot_labels=one_hot_labeled)
        masked_prediction_histories.append(masked_prediction_history)
    return full_prediction_history, masked_prediction_histories


def search_early_tickets(epochs, model_identifier, reset_epochs, pruning_percentages, pruning_iterations,
                         verbosity=2):
    print("Quantifying datapoints...")
    data_splits = mnist.quantify_datapoints()
    train_datapoints = data_splits['train']
    test_datapoints = data_splits['test']

    print("Developing full " + model_identifier + "...")
    base_model_wrapper = NetworkWrapper(model_identifier=model_identifier)
    # Read out the config for creation of the masked model
    base_model_config = base_model_wrapper.model.get_config()

    base_values = masking.extract_trainable_values(base_model_wrapper.model)

    list_of_intermediate_values = []
    histories_over_pruning_iterations = []
    for j in range(0, reset_epochs+1):
        list_of_intermediate_values.append(base_values)

    for i in range(0, pruning_iterations+1):
        iter_pruning_percentages = _iterate_pruning_percentages(percentages=pruning_percentages, iteration=i)
        _flag_pruning_iteration(i, iter_pruning_percentages)
        histories_over_reset_epochs = []
        for j in range(0, reset_epochs+1):
            _flag_reset_epoch(j)
            masked_model = masking.mask_model(trained_model=base_model_wrapper.model,
                                              initial_values=list_of_intermediate_values[j],
                                              pruning_percentages=iter_pruning_percentages)
            masked_wrapper = NetworkWrapper(model_identifier='GivenModel',
                                            given_model=masked_model)
            building_history = {'accuracy': [],
                                'precision': [],
                                'recall': [],
                                'confusion_matrices': []}
            for k in range(0, epochs):
                _flag_epoch(k + 1)
                last_history = inspect_metrics_while_training(model_wrapper=masked_wrapper,
                                                              training_data=train_datapoints,
                                                              validation_data=test_datapoints,
                                                              epochs=1,
                                                              batch_size=60,
                                                              verbosity=verbosity,
                                                              one_hot_labels=True,
                                                              flag_epochs=False)
                _extend_history(building_history, last_history)
                if k == j:
                    # copy weights by value to save them
                    list_of_intermediate_values[k] = masking.extract_trainable_values(masked_wrapper.model)
            # TODO: Does this work?
            histories_over_reset_epochs.append(building_history)
        histories_over_pruning_iterations.append(histories_over_reset_epochs)
    return histories_over_pruning_iterations

'''
---DEPRECATED---
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
    masked_model = masking.mask_model(trained_model=reuters_model_wrapper.model,
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
'''


def inspect_metrics_while_training(model_wrapper, training_data, validation_data, epochs, batch_size,
                                   verbosity=2, scheme_for_averaging="micro", one_hot_labels=False,
                                   flag_epochs=True):

    accuracy_over_epochs = []
    precision_over_epochs = []
    recall_over_epochs = []
    confusion_matrices = []
    if one_hot_labels:
        sparse_true_labels = _sparsify_predictions(validation_data[1])
    else:
        sparse_true_labels = validation_data[1]
    for i in range(epochs):
        if (verbosity > 0) & flag_epochs:
            _flag_epoch(i + 1)
        model_wrapper.train_model(training_data,
                                  batch_size=batch_size,
                                  epochs=1,
                                  verbosity=verbosity)

        sparse_predictions = _sparsify_predictions(model_wrapper.model.predict(validation_data))
        accuracy_over_epochs.append(sklearn.metrics.accuracy_score(sparse_true_labels, sparse_predictions))
        precision_over_epochs.append(sklearn.metrics.precision_score(sparse_true_labels,
                                                                     sparse_predictions,
                                                                     average=scheme_for_averaging))
        recall_over_epochs.append(sklearn.metrics.recall_score(sparse_true_labels,
                                                               sparse_predictions,
                                                               average=scheme_for_averaging))
        confusion_matrices.append(sklearn.metrics.confusion_matrix(sparse_true_labels, sparse_predictions))

    return {'accuracy': accuracy_over_epochs,
            'precision': precision_over_epochs,
            'recall': recall_over_epochs,
            'confusion_matrices': confusion_matrices}


def _extend_history(base_history, history_to_append):
    return {'accuracy': base_history['accuracy'].append(history_to_append['accuracy']),
            'precision': base_history['precision'].append(history_to_append['precision']),
            'recall': base_history['recall'].append(history_to_append['recall']),
            'confusion_matrices': base_history['confusion_matrices'].append(history_to_append['confusion_matrices'])
            }


def _sparsify_predictions(one_hot_predictions):
    return np.argmax(one_hot_predictions, axis=1)


def _flag_epoch(index_of_epoch):
    print(12 * "-" + "Epoch " + str(index_of_epoch) + 12 * "-")
    return


def _flag_pruning_iteration(index_of_pruning_iteration, pruning_percentage):
    print("-" * 18 + " Pruning Iteration " + str(index_of_pruning_iteration) + " " + "-" * 18)
    print("Total Pruning Percentage: " + str(pruning_percentage))
    return


def _flag_reset_epoch(index_of_reset_epoch):
    print("+" * 18 + " Train model for masking at epoch " + str(index_of_reset_epoch) + " " + "+" * 18)
    return


def _iterate_pruning_percentages(percentages, iteration):
    new_percentages = {}
    for key in percentages:
        new_percentages[key] = (1-np.power(1 - (percentages[key]/100), iteration))*100
    return new_percentages


