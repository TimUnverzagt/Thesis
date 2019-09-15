from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

# Personal modules
import preprocessing as prep
import custom_io as io
from network import CustomNetworkHandler as Network

# tf.debugging.set_log_device_placement(True)


def main():
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tok_train_docs, tok_test_docs] = io.load_corpus_docs()

    ## Use this code to setup your embedding for the first time
    # emb_dict = pre.prepare_embedding(tok_train_docs + tok_test_docs)
    # io.save_embedding(emb_dict)

    ## If you already have an embedding load it with this function
    emb_dict = io.load_embedding()

    # Embed the docs before you you feed them to the network
    emb_train_docs = prep.embed_docs(emb_dict, tok_train_docs)
    emb_test_docs = prep.embed_docs(emb_dict, tok_test_docs)
    (batched_train_words, batched_train_cats) = prep.batch_docs(emb_train_docs, target_doc_len=30)
    (batched_test_words, batched_test_cats) = prep.batch_docs(emb_test_docs, target_doc_len=30)

    model = Network(doc_len=30)

    model.train(input_array=batched_train_words,
                annotation_array=batched_train_cats)

    test_loss, test_acc = model.model.evaluate(batched_test_words, batched_test_cats)

    print("Loss: ", test_loss)
    print("Accuracy: ", test_acc)

    return


main()

