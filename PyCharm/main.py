from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf

# Personal modules
import preprocessing as pre
import custom_io as io


def main():
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tokenized_train_docs, tokenized_test_docs] = io.load_corpus()

    ## Use this code to setup your embedding for the first time
    # embedding_dict = pre.prepare_embedding(tokenized_train_docs + tokenized_test_docs)
    # io.save_embedding(embedding_dict)

    ## If you already have an embedding load it with this function
    # embedding_dict = io.load_embedding()
    return


main()

