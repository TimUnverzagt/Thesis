from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras.saving.saved_model import load as tfk_load

# Personal modules
import masking
from network import CustomNetworkWrapper as NetworkWrapper
from datasets import reuters
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
    return




main()
