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
    (batched_words, batched_cats) = batch_docs(emb_docs)

    model = tfk.Sequential([
        tfk.layers.Input(shape=(30, 300)),
        tfk.layers.Flatten(input_shape=(30, 300)),
        tfk.layers.Dense(9000, activation=tf.nn.relu),
        tfk.layers.Dense(900, activation=tf.nn.relu),
        tfk.layers.Dense(90, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(batched_words,
              batched_cats,
              batch_size=32,
              epochs=10)

    return


def batch_docs(emb_docs):
    # TODO: Implement
    no_of_docs = len(emb_docs)
    no_of_cats = len(io.load_corpus_categories())
    bat_words = np.zeros(shape=(no_of_docs, 30, 300))
    bat_cats = np.zeros(shape=(no_of_docs, no_of_cats))
    i = 0
    for index, doc in enumerate(emb_docs):
        no_words_in_doc = np.shape(doc[0])[0]

        # Gather embedding of words
        if no_words_in_doc >= 30:
            bat_words[index] = doc[0][0:30]
        else:
            i += 1
            # Pad documents that are too short (atm implicit zero-padding)
            bat_words[index][0:no_words_in_doc] = doc[0][:]

        # Gather embedding of categories
        bat_cats[index] = doc[2]

    return bat_words, bat_cats


main()

