from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import datetime

# Personal modules
import storage
import experiments
import visualization
from datasets import newsgroups

# tf.debugging.set_log_device_placement(True)


def main():
    # For debugging
    # tf.config.experimental_run_functions_eagerly(True)

    # Hack to prevent a specific error with cudNN
    # https://github.com/tensorflow/tensorflow/issues/24828
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    histories_path = '../PyCharm/Histories'


    # task_description = 'Transfer'
    # task_description = 'Reproduction'
    task_description = 'Early-Tickets'
    # architecture_description = 'Newsgroups-End2End-CNN'
    # architecture_description = 'CIFAR10-CNN-6'
    architecture_description = 'MNIST-Lenet-FCN'

    pruning_percentages = {'dense': 20, 'conv': 15}

    searching_for_early_tickets = (task_description == 'Early-Tickets')

    # Set parameters specific to certain architectures
    if architecture_description == 'MNIST-Lenet-FCN':
        pruning_percentages = {'dense': 20, 'conv': 0}
        architecture_verbosity = 2
        if searching_for_early_tickets:
            approx_no_epochs_needed_for_convergence = 15
        else:
            approx_no_epochs_needed_for_convergence = 50
    elif architecture_description == 'CIFAR10-CNN-6':
        approx_no_epochs_needed_for_convergence = 36
        architecture_verbosity = 1
    elif architecture_description == 'Newsgroups-End2End-CNN':
        approx_no_epochs_needed_for_convergence = 10
        architecture_verbosity = 1

    execution_date = str(datetime.date.today())

    train = False
    # visualize = False
    visualize = not train
    test_new_structure = False

    if train:
        experiment_path = histories_path + \
                          '/' + \
                          task_description + \
                          '/' + \
                          architecture_description + \
                          '/' + \
                          execution_date
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        os.mkdir(experiment_path)
        for i in range(0, 1):
            folder_path = experiment_path + \
                          '/' + \
                          str(i)
            os.mkdir(folder_path)

            if searching_for_early_tickets:
                histories_over_pruning_iterations = \
                    experiments.search_early_tickets(
                        epochs=approx_no_epochs_needed_for_convergence,
                        model_identifier=architecture_description,
                        reset_epochs=approx_no_epochs_needed_for_convergence,
                        pruning_percentages=pruning_percentages,
                        pruning_iterations=10,
                        verbosity=architecture_verbosity
                    )

                storage.save_experimental_history(histories_over_pruning_iterations[0], path=folder_path, name='full')
                for idx, masked_network_history in enumerate(histories_over_pruning_iterations[1:]):
                    model_name = 'masked_' + \
                                 str(pruning_percentages['dense']) + \
                                 '_times_' + \
                                 str(idx+1)
                    storage.save_experimental_history(masked_network_history, path=folder_path, name=model_name)

            else:
                (full_network_history, masked_network_histories) = \
                    experiments.search_lottery_tickets(
                        epochs=approx_no_epochs_needed_for_convergence,
                        model_identifier=architecture_description,
                        pruning_percentages=pruning_percentages,
                        pruning_iterations=25,
                        verbosity=architecture_verbosity)

                storage.save_experimental_history(full_network_history, path=folder_path, name='full')
                for idx, masked_network_history in enumerate(masked_network_histories):
                    model_name = 'masked_' + \
                                 str(pruning_percentages['dense']) + \
                                 '|' + \
                                 str(pruning_percentages['conv']) + \
                                 '_times_' + \
                                 str(idx + 1)
                    storage.save_experimental_history(masked_network_history, path=folder_path, name=model_name)

    if visualize:
        # TODO: Add readout for early-tick-search
        folder_path = histories_path + \
                      '/Visualization/' + \
                      task_description + \
                      '/' + \
                      architecture_description + \
                      '/' + \
                      str(0)

        full_network_history = storage.load_experimental_history(path=folder_path, name='full')

        no_interation_to_compare = 5

        if architecture_description == 'MNIST-Lenet-FCN':
            history_name = 'masked_' + \
                           str(pruning_percentages['dense']) + \
                           '_times_' + \
                           str(no_interation_to_compare)

        else:
            history_name = 'masked_' + \
                           str(pruning_percentages['dense']) + \
                           '|' + \
                           str(pruning_percentages['conv']) + \
                           '_times_' + \
                           str(no_interation_to_compare)

        if searching_for_early_tickets:
            network_histories_per_pruning_iteration = storage.load_experimental_history(path=folder_path, name=history_name)

            print("There is no visualization for early-ticket search yet...")

        else:
            masked_network_history = storage.load_experimental_history(path=folder_path, name=history_name)

            visualization.plot_measure_comparision_over_training(full_network_history, 'Full Network',
                                                                masked_network_history, 'Masked Network',
                                                                'accuracy', 'accuracy')

    if test_new_structure:
        experiments.test_cnn_for_nlp(epochs=5)
        # print("No new structure to test. Check main.py")

    return


main()
