from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf

# Personal modules
import preprocessing as prep
import custom_io as io


def main():
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tokenized_train_docs, tokenized_test_docs] = io.load_corpus()

    ## Use this code to setup your embedding for the first time
    # embedding_dict = pre.prepare_embedding(tokenized_train_docs + tokenized_test_docs)
    # io.save_embedding(embedding_dict)

    ## If you already have an embedding load it with this function
    embedding_dict = io.load_embedding()

    # Embed the docs before you you feed them to the network
    embedded_docs = prep.embed_docs(embedding_dict, tokenized_train_docs)
    # print(type(embedded_docs[0]))
    # print(type(embedded_docs[0][0]))
    # print(type(embedded_docs[0][0][0]))
    # print(type(embedded_docs[0][1]))
    # print(type(embedded_docs[0][1][0]))
    # print(type(embedded_docs[0][1][0][0]))
    #
    # print(np.shape(embedded_docs[0][0]))
    # print(np.shape(embedded_docs[0][1][0]))

    return


main()

