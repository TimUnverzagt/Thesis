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
    # task_description = 'Reroduction'
    task_description = 'Transfer'
    # architecture_description = 'CIFAR10-CNN-6'
    # architecture_description = 'MNIST-Lenet-FCN'
    architecture_description = 'Newsgroups-End2End-CNN'
    pruning_percentages = {'dense': 20,
                           'conv': 15}
    execution_date = str(datetime.date.today())
    # experiment_path = histories_path + '/' + task_description + '/' + architecture_description + '-Test' + '/' + execution_date
    experiment_path = histories_path + '/' + task_description + '/' + architecture_description + '/' + execution_date

    train = True
    visualize = False
    # visualize = not train
    test_new_structure = False
    if train:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
        os.mkdir(experiment_path)
        for i in range(0, 1):
            folder_path = experiment_path + '/' + str(i)
            os.mkdir(folder_path)

            '''
            histories_over_pruning_iterations = \
            '''
            (full_network_history, masked_network_histories) = \
                experiments.search_lottery_tickets(epochs=0,
                                                   model_identifier=architecture_description,
                                                   pruning_percentages=pruning_percentages,
                                                   pruning_iterations=1,
                                                   verbosity=1)

            storage.save_experimental_history(full_network_history, path=folder_path, name='full')
            for idx, masked_network_history in enumerate(masked_network_histories):
                model_name = 'masked_' + str(pruning_percentages['dense']) + '|' + str(pruning_percentages['conv']) +\
                             '_times_' + str(idx+1)
                storage.save_experimental_history(masked_network_history, path=folder_path, name=model_name)
            '''

            storage.save_experimental_history(histories_over_pruning_iterations[0], path=folder_path, name='full')
            for idx, masked_network_history in enumerate(histories_over_pruning_iterations[1:]):
                model_name = 'masked_' + str(pruning_percentages) + '_times_' + str(idx+1)
                storage.save_experimental_history(masked_network_history, path=folder_path, name=model_name)
            '''

    if visualize:
        # TODO: Add readout for early-tick-search
        folder_path = experiment_path + '/' + str(0)
        full_network_history = storage.load_experimental_history(path=folder_path, name='full')
        masked_network_history = storage.load_experimental_history(path=folder_path, name='full')
        '''
        masked_network_history = \
            storage.load_experimental_history(
                path=folder_path,
                name='masked_' + str(pruning_percentages['dense']) + '|' + str(pruning_percentages['conv']) +
                     '_times_7')
        '''

        visualization.plot_measure_comparision_over_training(full_network_history, 'Full Network',
                                                             masked_network_history, 'Masked Network',
                                                             'accuracy', 'accuracy')
    if test_new_structure:
        experiments.test_cnn_for_nlp(epochs=5)
        # print("No new structure to test. Check main.py")

    return


main()
