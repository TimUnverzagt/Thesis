from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


def mask_network():
    init_model = tfk.models.load_model('SavedModels/test-init')
    init_model.summary()
    trained_model = tfk.models.load_model('SavedModels/test-trained')
    trained_model.summary()
    return
