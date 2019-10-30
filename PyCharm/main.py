from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras.saving.saved_model import load as tfk_load
from pathlib import Path

# Personal modules
import preprocessing as prep
import custom_io as io
import masking
from network import CustomNetworkHandler as Network
from custom_layers import MaskedDense
# tf.debugging.set_log_device_placement(True)


def main():
    # For debugging
    tf.config.experimental_run_functions_eagerly(True)

    # Hack to prevent a specific error with cudNN
    # https://github.com/tensorflow/tensorflow/issues/24828
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    reuters_model_handler = Network(no_of_features=0,
                                    model_name='GivenModel',
                                    given_model=tfk.models.load_model('SavedModels/test-trained'))

    # reuters_model.save_model_as_file('test-trained')

    (train_datapoints, test_datapoints) = construct_features_for_reuters(target_no_of_features=30)

    lottery_ticket = construct_lottery_ticket(trained_model=reuters_model_handler.model,
                                              init_model=tfk.models.load_model('SavedModels/test-init'))
    lottery_ticket_handler = Network(no_of_features=0,
                                     model_name='GivenModel',
                                     given_model=lottery_ticket)
    lottery_ticket_handler.train(train_datapoints)

    evaluate_model(reuters_model_handler.model, test_datapoints)
    evaluate_model(lottery_ticket_handler.model, test_datapoints)

    return


def construct_lottery_ticket(trained_model, init_model):
    masks = masking.create_masks(trained_model)

    init_model.summary()
    model_config = init_model.get_config()

    masked_model = tfk.Sequential()
    for idx, layer in enumerate(init_model.layers):
        print(model_config['layers'][idx]['class_name'])
        if model_config['layers'][idx]['class_name'] == 'Dense':
            print("Replacing Dense-layer of the model with a custom MaskedDense-layer")
            if model_config['layers'][idx]['config']['activation'] == 'relu':
                print("Recognized an relu-activation.")
                old_activation = tf.nn.relu
            elif model_config['layers'][idx]['config']['activation'] == 'sigmoid':
                print("Recognized an sigmoid-activation.")
                old_activation = tf.nn.sigmoid
            else:
                # TODO: Throw real exception
                print('The activation of the given model is not recognized.')
                print('No activation was chosen. This will likely result in a critical error!')

            replacement_layer = MaskedDense(units=layer.output_shape[1],
                                            activation=old_activation,
                                            kernel=layer.kernel,
                                            mask=masks[idx],
                                            bias=layer.bias
                                            )
            masked_model.add(replacement_layer)
        else:
            masked_model.add(layer)

    masked_model.build()
    masked_model.summary()
    return masked_model


def construct_features_for_reuters(target_no_of_features):
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tok_train_docs, tok_test_docs] = io.load_corpus_docs()

    if Path("dictionary.p").exists():
        print("Loading previous language model...")
        emb_dict = io.load_embedding()
    else:
        print("No previous model found!")
        print("Beginning to construct new model with GoogleNews-Vectors as base...")
        emb_dict = prep.prepare_embedding(tok_train_docs + tok_test_docs)
        io.save_embedding(emb_dict)

    # Embed the docs before you you feed them to the network
    print("Embedding documents...")
    emb_train_docs = prep.embed_docs(emb_dict, tok_train_docs)
    emb_test_docs = prep.embed_docs(emb_dict, tok_test_docs)
    (batched_train_words, batched_train_cats) = prep.batch_docs(emb_train_docs, target_doc_len=target_no_of_features)
    (batched_test_words, batched_test_cats) = prep.batch_docs(emb_test_docs, target_doc_len=target_no_of_features)
    return ((batched_train_words, batched_train_cats),
            (batched_test_words, batched_test_cats))


def construct_model_handler_for_reuters():
    ((batched_train_words, batched_train_cats),
     (batched_test_words, batched_test_cats)) = construct_features_for_reuters(target_no_of_features=30)

    print("Developing network...")
    model_handler = Network(no_of_features=30,
                            model_name='FeedForward')
    # Add a channel dimension for CNNs
    # batched_train_words = np.reshape(batched_train_words, np.shape(batched_train_words) + (1,))
    # batched_test_words = np.reshape(batched_test_words, np.shape(batched_test_words) + (1,))

    # model_handler.save_model_as_file('test-init')

    print("Training network...")
    model_handler.train(datapoints=(batched_train_words, batched_train_cats))

    return model_handler


def evaluate_model(model, datapoints):
    test_loss, test_recall, test_precision = model.evaluate(datapoints[0], datapoints[1])

    print("Loss: ", test_loss)
    print("Recall: ", test_recall)
    print("Precision: ", test_precision)
    if not (test_precision + test_recall) == 0:
        f1_measure = 2 * (test_precision*test_recall) / (test_precision + test_recall)
    else:
        f1_measure = 0

    print("F1: ", f1_measure)
    return


main()
