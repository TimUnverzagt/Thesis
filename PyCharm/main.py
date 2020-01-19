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

    train = False
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
        approx_no_epochs_needed_for_convergence = 50
        no_pruning_iterations = 25
        if searching_for_early_tickets:
            no_epochs_considered_for_masking = 10
    elif architecture_description == 'CIFAR10-CNN-6':
        approx_no_epochs_needed_for_convergence = 36
        architecture_verbosity = 1
        no_pruning_iterations = 25
    elif architecture_description == 'Newsgroups-End2End-CNN':
        # approx_no_epochs_needed_for_convergence = 10
        approx_no_epochs_needed_for_convergence = 10
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
                        reset_epochs=no_epochs_considered_for_masking,
                        pruning_percentages=pruning_percentages,
                        pruning_iterations=no_pruning_iterations,
                        verbosity=architecture_verbosity
                    )

                for idx, masked_network_history in enumerate(histories_over_pruning_iterations):
                    model_name = 'masked_' + \
                                 str(pruning_percentages['dense']) + \
                                 '|' + \
                                 str(pruning_percentages['conv']) + \
                                 '_at_epoch_' + \
                                 str(idx)
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
        # no_masking_iteration_provided = 10
        no_masking_iteration_provided = no_pruning_iterations
        measure_key = 'accuracy'
        # The convergence of the full network happens until:
        # epoch 10 (Reproduction-MNIST-Lenet-FCN)
        # epoch 5 (Reproduction-CIFAR10-CNN-6)
        # epoch 3 (Transfer-20Newsgroups-End2EndCNN)
        no_epochs_to_estimated_convergence = 3
        # The effective amount of weights remaining per iteration
        # 0.8 (Reproduction-MNIST-Lenet-FCN)
        # 0.8253 (Reproduction-MNIST-Lenet-FCN)
        # 0.80005 (Transfer-20Newsgroups-End2EndCNN)
        compound_remaining_size = 0.80005
        # Limits of the axis for the experiment:
        # y_limits = (0.6, 0.85)
        # y_limits = (0.6, 0.8)
        y_limits = (0.9, 1.0)
        x_limits = (0, no_masking_iteration_provided)

        # no_experiments_provided = 3
        average_over_multiple_experiments = False
        if average_over_multiple_experiments:
            # This part of the code was not used to produce any results in the corresponding thesis to this framework
            # Because the experiments proved to behave deterministically even tho the input data was shuffled.
            # I am uncertain how that could happen, but it resulted in an inability to meaningfully average experiments
            """
            experiment_results = []
            if searching_for_early_tickets:
                print("There is no visualization for multiple early-ticket searches yet...")
                for i in range(no_experiments_provided):
                    folder_path = histories_path + '/' + \
                                  'Visualization/' + \
                                  task_description + '/' + \
                                  architecture_description + '/' + \
                                  str(0)
                    pruning_names = []
                    pruning_results = []

                    for j in range(no_masking_iteration_provided+1):
                        pruning_name = 'masked_' + \
                                        str(pruning_percentages['dense']) + \
                                        '|' + \
                                        str(pruning_percentages['conv']) + \
                                        '_at_epoch_' + \
                                        str(j + 1)

                        pruning_results.append(
                            storage.load_experimental_history(path=folder_path, name=pruning_name)
                        )
                        pruning_names.append('after Epoch ' + str(j + 1))

                    experiment_results.append({'pruning_results': pruning_results,
                                                'pruning_names': pruning_names})

                visualization.plot_averaged_early_tickets(experiment_results, 'accuracy')

            else:
                for i in range(no_experiments_provided):
                    folder_path = histories_path + '/' + \
                                  'Visualization/' + \
                                  task_description + '/' + \
                                  architecture_description + '/' + \
                                  str(0)
                    network_names = []
                    network_history_dicts = []

                    network_names.append('Full Network')
                    network_history_dicts.append(storage.load_experimental_history(path=folder_path, name='full'))

                    # for j in range(10):
                    for j in range(no_masking_iteration_provided):
                        history_name = 'masked_' + str(pruning_percentages['dense'])
                        history_name = history_name + '|' + str(pruning_percentages['conv'])
                        history_name = history_name + '_times_' + str(j + 1)

                        network_history_dicts.append(
                            storage.load_experimental_history(path=folder_path, name=history_name)
                        )
                        network_names.append(str(j+1) + 'x Masked Network')
                    experiment_results.append({'network_history_dicts': network_history_dicts,
                                        'network_names': network_names})

                visualization.plot_averaged_experiments(experiment_results, measure_key)
            """

        else:
            folder_path = histories_path + '/' + \
                         'Visualization/' + \
                         task_description + '/' + \
                         architecture_description + '/' + \
                         str(0)
            if searching_for_early_tickets:
                pruning_names = []
                pruning_results = []

                for i in range(no_epochs_considered_for_masking+1):
                    pruning_name = 'masked_' + \
                                    str(pruning_percentages['dense']) + \
                                    '|' + \
                                    str(pruning_percentages['conv']) + \
                                    '_at_epoch_' + \
                                    str(i)

                    pruning_results.append(
                        storage.load_experimental_history(path=folder_path, name=pruning_name)
                    )
                    pruning_names.append('after Epoch ' + str(i))
                pruning_results.append(storage.load_experimental_history(
                    path=folder_path,
                    name='masked_20|0_at_final_epoch'
                ))
                pruning_names.append('after final Epoch')

                vis_results = []
                vis_names = []
                vis_results.append(pruning_results[0])
                vis_names.append(pruning_names[0])
                vis_results.append(pruning_results[1])
                vis_names.append(pruning_names[1])
                vis_results.append(pruning_results[2])
                vis_names.append(pruning_names[2])
                vis_results.append(pruning_results[3])
                vis_names.append(pruning_names[3])
                vis_results.append(pruning_results[4])
                vis_names.append(pruning_names[4])
                vis_results.append(pruning_results[5])
                vis_names.append(pruning_names[5])
                vis_results.append(pruning_results[-1])
                vis_names.append(pruning_names[-1])

                # colors = ["royalblue", "green", "peru", "crimson"]
                # colors = ["mediumpurple", "saddlebrown", "crimson"]
                colors = ["royalblue", "green", "peru", "mediumpurple", "saddlebrown", "gold", "crimson"]
                #"""
                visualization.plot_converged_measure_over_different_pruning_depths(
                    vis_results,
                    vis_names,
                    'accuracy',
                    head_length_to_truncate=no_epochs_to_estimated_convergence,
                    y_limits=y_limits,
                    colors=colors
                )
                #"""
                """
                visualization.plot_measure_over_n_trainings(
                    histories=pruning_results[2],
                    history_names=pruning_names,
                    measure_key=measure_key
                )
                """

            else:
                network_names = []
                network_history_dicts = []
                network_names.append('Full Network')
                network_history_dicts.append(storage.load_experimental_history(path=folder_path, name='full'))

                # for i in range(10):
                for i in range(no_masking_iteration_provided):
                    history_name = 'masked_' + str(pruning_percentages['dense'])
                    history_name = history_name + '|' + str(pruning_percentages['conv'])
                    history_name = history_name + '_times_' + str(i + 1)

                    network_history_dicts.append(
                        storage.load_experimental_history(path=folder_path, name=history_name)
                    )
                    network_names.append('Network pruned ' + str(i+1) + ' times')
                """
                vis_dicts = []
                vis_names = []
                colors = ["royalblue", "darkorange", "green"]
                # colors = ["royalblue", "green", "crimson", "mediumpurple", "saddlebrown"]
                vis_dicts.append(network_history_dicts[0])
                vis_names.append(network_names[0])
                vis_dicts.append(network_history_dicts[3])
                vis_names.append(network_names[3])
                vis_dicts.append(network_history_dicts[7])
                vis_names.append(network_names[7])
                #vis_dicts.append(network_history_dicts[12])
                #vis_names.append(network_names[12])
                #vis_dicts.append(network_history_dicts[15])
                #vis_names.append(network_names[15])
                #vis_dicts.append(network_history_dicts[18])
                #vis_names.append(network_names[18])

                visualization.plot_measure_over_n_trainings(
                    histories=vis_dicts,
                    history_names=vis_names,
                    measure_key=measure_key,
                    y_limits=y_limits,
                    colors=colors
                )
                """
                #"""
                visualization.plot_measure_over_n_trainings(
                    histories=network_history_dicts,
                    history_names=network_names,
                    measure_key=measure_key,
                    y_limits=y_limits
                )
                #"""
                """
                visualization.plot_average_measure_after_convergence(
                    experiment_result=network_history_dicts,
                    history_names=network_names,
                    measure_key=measure_key,
                    head_length_to_truncate=no_epochs_to_estimated_convergence,
                    y_limits=y_limits,
                    x_limits=x_limits,
                    remaining_ratio=compound_remaining_size
                )
                """

    return


main()
