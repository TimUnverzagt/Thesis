from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
import matplotlib.pyplot as plt

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

    (full_network_history_wrapper, masked_network_history_wrapper) = experiments.test_basic_network_of_the_paper(epochs=100)

    # storage.save_experimental_history(full_network_history_wrapper.history, name='full_training_with_validation_0')
    # storage.save_experimental_history(masked_network_history_wrapper.history, name='masked_training_with_validation_0')

    # folderpath = 'Lenet-FCN-CCE/test'
    # full_network_history_wrapper = storage.load_experimental_history('full_training_with_validation_0', folder=folderpath)
    # masked_network_history_wrapper = storage.load_experimental_history('masked_training_with_validation_0', folder=folderpath)

    visualization.plot_measure_comparision_over_training(full_network_history_wrapper.history, 'Full Network',
                                                         masked_network_history_wrapper.history, 'Masked Network',
                                                         'accuracy', 'accuracy')

    return


main()
