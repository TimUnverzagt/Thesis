# General modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def quantify_datapoints():
    cifar_datapoints = cifar10.load_data()
    print("")

    # quantify the categories (for the training images)
    training_datapoints = cifar_datapoints[0]
    no_of_training_images = training_datapoints[1].shape[0]
    quantified_categories = np.zeros(shape=(no_of_training_images, 10))
    # TODO: Can I use something less convoluted than np.ndenumerate here?
    for idx, cat in np.ndenumerate(training_datapoints[1]):
        quantified_categories[idx[0]][cat] = 1
    training_datapoints = (training_datapoints[0].astype(dtype=np.float16),
                           quantified_categories.astype(dtype=np.float16))

    # quantify the categories (for the test images)
    test_datapoints = cifar_datapoints[1]
    no_of_test_images = test_datapoints[1].shape[0]
    quantified_categories = np.zeros(shape=(no_of_test_images, 10))
    for idx, cat in np.ndenumerate(test_datapoints[1]):
        quantified_categories[idx[0]][cat] = 1
    test_datapoints = (test_datapoints[0].astype(dtype=np.float16),
                       quantified_categories.astype(dtype=np.float16))

    datapoints = {'train': training_datapoints, 'test': test_datapoints}
    return datapoints

