from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

# Personal modules
import preprocessing as prep
import custom_io as io
from network import CustomNetworkHandler as Network


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
    (batched_words, batched_cats) = prep.batch_docs(emb_docs, target_doc_len=30)

    model = Network(doc_len=30)

    model.train(input_array=batched_words,
                annotation_array=batched_cats)

    return


main()

