from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import tensorflow as tf
import matplotlib.pyplot as plt

# Personal modules
import experiments
# tf.debugging.set_log_device_placement(True)


def main():
    # For debugging
    # tf.config.experimental_run_functions_eagerly(True)

    # Hack to prevent a specific error with cudNN
    # https://github.com/tensorflow/tensorflow/issues/24828
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    (full_network_history, masked_network_history) = experiments.test_creation_of_masked_network(20)
    full_network_recall = full_network_history.history['recall']
    masked_network_recall = masked_network_history.history['recall_1']
    epoch_count = range(1, len(full_network_recall) + 1)

    plt.plot(epoch_count, full_network_recall, 'r--')
    plt.plot(epoch_count, masked_network_recall, 'b-')
    plt.legend(['Original Recall', 'Recall after Masking'])
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.show()
    return


main()
