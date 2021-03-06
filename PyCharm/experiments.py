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


def search_lottery_tickets(epochs, model_identifier, pruning_percentages, pruning_iterations, verbosity=2):
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
    # Shuffle the before each experiment
    print("Shuffleling the datapoints...")
    train_datapoints = _shuffle_in_unison(train_datapoints[0], train_datapoints[1])
    test_datapoints = _shuffle_in_unison(test_datapoints[0], test_datapoints[1])

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
    base_wrapper = NetworkWrapper(model_identifier=model_identifier)

    base_values = masking.extract_trainable_values(base_wrapper.model)

    values_of_previous_training = []
    for i in range(0, reset_epochs+1):
        values_of_previous_training.append([base_values])

    pruning_results = []
    previous_wrapper = base_wrapper
    for i in range(0, reset_epochs + 1):
        _flag_reset_epoch(i)
        histories = []
        for j in range(0, pruning_iterations + 1):
            iter_pruning_percentages = _iterate_pruning_percentages(percentages=pruning_percentages, iteration=j)
            _flag_pruning_iteration(j, iter_pruning_percentages)
            masked_model = masking.mask_model(trained_model=previous_wrapper.model,
                                              initial_values=values_of_previous_training[i][j],
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
                    # copy weights by value to save them for the next pruning iteration
                    values_of_previous_training[i].append(masking.extract_trainable_values(masked_wrapper.model))

            previous_wrapper = masked_wrapper
            histories.append(building_history)
        pruning_results.append(histories)
    return pruning_results


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
    print("")
    print(58*"=")
    print("=" * 18 + " Pruning Iteration " + str(index_of_pruning_iteration) + " " + "=" * 18)
    print("Total Pruning Percentage: " + str(pruning_percentage))
    print(58*"=")
    print("")
    return


def _flag_reset_epoch(index_of_reset_epoch):
    print("+" * 18 + " Train model for masking at epoch " + str(index_of_reset_epoch) + " " + "+" * 18)
    return


def _iterate_pruning_percentages(percentages, iteration):
    new_percentages = {}
    for key in percentages:
        new_percentages[key] = (1-np.power(1 - (percentages[key]/100), iteration))*100
    return new_percentages


# Credit to Íhor Mé
# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def _shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


