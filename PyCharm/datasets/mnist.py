# General modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def quantify_datapoints():
    mnist_datapoints = mnist.load_data()

    # quantify the categories (for the training images)
    training_datapoints = mnist_datapoints[0]
    no_of_training_images = training_datapoints[1].shape[0]
    quantified_categories = np.zeros(shape=(no_of_training_images, 10))
    for idx, cat in np.ndenumerate(training_datapoints[1]):
        quantified_categories[idx][cat] = 1
    training_datapoints = (training_datapoints[0], quantified_categories)

    # quantify the categories (for the test images)
    test_datapoints = mnist_datapoints[1]
    no_of_test_images = test_datapoints[1].shape[0]
    quantified_categories = np.zeros(shape=(no_of_test_images, 10))
    for idx, cat in np.ndenumerate(test_datapoints[1]):
        quantified_categories[idx][cat] = 1
    test_datapoints = (test_datapoints[0], quantified_categories)

    datapoints = {'train': training_datapoints, 'test': test_datapoints}
    return

