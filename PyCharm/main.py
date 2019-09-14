from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

# Personal modules
import preprocessing as prep
import custom_io as io


def main():
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tok_train_docs, tok_test_docs] = io.load_corpus_docs()

    ## Use this code to setup your embedding for the first time
    # emb_dict = pre.prepare_embedding(tok_train_docs + tok_test_docs)
    # io.save_embedding(emb_dict)

    ## If you already have an embedding load it with this function
    emb_dict = io.load_embedding()

    # Embed the docs before you you feed them to the network
    emb_docs = prep.embed_docs(emb_dict, tok_train_docs)
    print(np.shape(emb_docs[0][0]))

    model = tfk.Sequential([
        tfk.layers.Flatten(input_shape=(30, 300)),
        tfk.layers.Dense(9000, activation=tf.nn.relu),
        tfk.layers.Dense(900, activation=tf.nn.relu),
        tfk.layers.Dense(90, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return


def batch_docs(emb_docs):
    # TODO: Implement
    batched_emb_shape = (30, 300)
    for doc in emb_docs:
        print()

    return


main()

