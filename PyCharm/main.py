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

    (full_network_history, masked_network_history) = experiments.test_basic_network_of_the_paper(epochs=10)
    # (full_network_history, masked_network_history) = experiments.test_creation_of_masked_network(epochs=3)

    storage.save_experimental_history(full_network_history.history, name='full_network_training_test')
    storage.save_experimental_history(masked_network_history.history, name='masked_network_training_test')

    visualization.plot_measure_comparision_over_training(full_network_history, 'Full Network',
                                                         masked_network_history, 'Masked Network',
                                                         'loss')

    return


main()
