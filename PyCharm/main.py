from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime

# Personal modules
import storage
import experiments
import visualization

# tf.debugging.set_log_device_placement(True)


def main():
    # For debugging
    # tf.config.experimental_run_functions_eagerly(True)

    # Hack to prevent a specific error with cudNN
    # https://github.com/tensorflow/tensorflow/issues/24828
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    histories_path = '../PyCharm/Histories'
    task_description = 'Reproduction'
    architecture_description = 'MNIST-Lenet-FCN'
    pruning_percentage = 20
    execution_date = str(datetime.date.today())
    experiment_path = histories_path + '/' + task_description + '/' + architecture_description + '-Iter' + '/' + execution_date

    os.mkdir(experiment_path)
    for i in range(0, 10):
        folder_path = experiment_path + '/' + str(i)
        os.mkdir(folder_path)

        (full_network_history, masked_network_histories) = \
            experiments.search_for_lottery_tickets(epochs=5,
                                                   model_identifier=architecture_description,
                                                   pruning_percentage=pruning_percentage,
                                                   pruning_iterations=3)

        storage.save_experimental_history(full_network_history, path=folder_path, name='full')
        for idx, masked_network_history in enumerate(masked_network_histories):
            model_name = 'full_' + str(pruning_percentage) + 'iter_' + str(idx)
            storage.save_experimental_history(masked_network_history, path=folder_path, name=model_name)

    # full_network_history = storage.load_experimental_history(path=folder_path, name='full_training_20')
    # masked_network_history = storage.load_experimental_history(path=folder_path, name='full_training_20')

    # visualization.plot_measure_comparision_over_training(full_network_history, 'Full Network',
    #                                                      masked_network_history, 'Masked Network',
    #                                                      'accuracy', 'accuracy')

    return


main()
