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

    train = True
    # visualize = False
    visualize = not train

    no_experiments = 1

    task_description = 'Transfer'
    # task_description = 'Reproduction'
    # task_description = 'Early-Tickets'

    # architecture_description = 'CIFAR10-CNN-6'
    # architecture_description = 'MNIST-Lenet-FCN'
    # Only compatible with 'Transfer'
    architecture_description = 'Newsgroups-End2End-CNN'

    pruning_percentages = {'dense': 20, 'conv': 15}

    searching_for_early_tickets = (task_description == 'Early-Tickets')

    # Set parameters specific to certain architectures
    if architecture_description == 'MNIST-Lenet-FCN':
        pruning_percentages = {'dense': 20, 'conv': 0}
        architecture_verbosity = 2
        if searching_for_early_tickets:
            approx_no_epochs_needed_for_convergence = 15
            no_pruning_iterations = 15
        else:
            approx_no_epochs_needed_for_convergence = 50
            no_pruning_iterations = 25
    elif architecture_description == 'CIFAR10-CNN-6':
        approx_no_epochs_needed_for_convergence = 36
        architecture_verbosity = 1
        no_pruning_iterations = 25
    elif architecture_description == 'Newsgroups-End2End-CNN':
        # approx_no_epochs_needed_for_convergence = 10
        approx_no_epochs_needed_for_convergence = 1
        architecture_verbosity = 1
        no_pruning_iterations = 10

    execution_date = str(datetime.date.today())

    if train:
        experiment_path = histories_path + \
                          '/' + \
                          task_description + \
                          '/' + \
                          architecture_description + \
                          '/' + \
                          execution_date
        if os.path.exists(experiment_path):
            print("Experiment-directory already exists")
            return
            # shutil.rmtree(experiment_path)
        os.mkdir(experiment_path)
        for i in range(0, no_experiments):
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
                        pruning_iterations=no_pruning_iterations,
                        verbosity=architecture_verbosity
                    )

                storage.save_experimental_history(histories_over_pruning_iterations[0], path=folder_path, name='full')
                for idx, masked_network_history in enumerate(histories_over_pruning_iterations[1:]):
                    model_name = 'masked_' + \
                                 str(pruning_percentages['dense']) + \
                                 '_times_' + \
                                 str(idx + 1)
                    storage.save_experimental_history(masked_network_history, path=folder_path, name=model_name)

            else:
                (full_network_history, masked_network_histories) = \
                    experiments.search_lottery_tickets(
                        epochs=approx_no_epochs_needed_for_convergence,
                        model_identifier=architecture_description,
                        pruning_percentages=pruning_percentages,
                        pruning_iterations=no_pruning_iterations,
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
        # Values to be set manually
        no_masking_iteration_provided = 25
        average_over_multiple_experiments = True
        no_experiments_provided = 3
        measure_key = 'accuracy'
        # TODO: Add readout for early-tick-search
        if average_over_multiple_experiments:
            print("Averaging not yet supported")
            if searching_for_early_tickets:
                print("There is no visualization for early-ticket search yet...")

            else:
                experiment_results = []
                for i in range(no_experiments_provided):
                    folder_path = histories_path + '/' + \
                                  'Visualization/' + \
                                  task_description + '/' + \
                                  architecture_description + '/' + \
                                  str(0)
                    network_names = []
                    network_histories = []

                    network_names.append('Full Network')
                    network_histories.append(storage.load_experimental_history(path=folder_path, name='full'))

                    # for j in range(10):
                    for j in range(no_masking_iteration_provided):
                        history_name = 'masked_' + str(pruning_percentages['dense'])
                        history_name = history_name + '|' + str(pruning_percentages['conv'])
                        history_name = history_name + '_times_' + str(j + 1)

                        network_histories.append(
                            storage.load_experimental_history(path=folder_path, name=history_name)
                        )
                        network_names.append(str(j+1) + 'x Masked Network')
                    experiment_results.append({'network_histories': network_histories,
                                        'network_names': network_names})

                visualization.plot_averaged_experiments(experiment_results, measure_key)

        else:
            if searching_for_early_tickets:
                print("There is no visualization for early-ticket search yet...")

            else:
                folder_path = histories_path + '/' + \
                              'Visualization/' + \
                              task_description + '/' + \
                              architecture_description + '/' + \
                              str(0)
                network_names = []
                network_histories = []

                network_names.append('Full Network')
                network_histories.append(storage.load_experimental_history(path=folder_path, name='full'))

                # for i in range(10):
                for i in range(no_masking_iteration_provided):
                    history_name = 'masked_' + str(pruning_percentages['dense'])
                    history_name = history_name + '|' + str(pruning_percentages['conv'])
                    history_name = history_name + '_times_' + str(i + 1)

                    network_histories.append(
                        storage.load_experimental_history(path=folder_path, name=history_name)
                    )
                    network_names.append(str(i+1) + 'x Masked Network')

                visualization.plot_measure_over_n_trainings(
                    histories=network_histories,
                    history_names=network_names,
                    measure_key=measure_key
                )

    return


main()
