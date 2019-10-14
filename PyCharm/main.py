from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
from pathlib import Path

# Personal modules
import preprocessing as prep
import custom_io as io
import masking
from network import CustomNetworkHandler as Network

# tf.debugging.set_log_device_placement(True)


def main():
    # train_a_network()
    masking.mask_network()
    return


def train_a_network():
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
    (batched_train_words, batched_train_cats) = prep.batch_docs(emb_train_docs, target_doc_len=30)
    (batched_test_words, batched_test_cats) = prep.batch_docs(emb_test_docs, target_doc_len=30)

    print("Developing network...")
    model = Network(target_doc_len=30, model_name='sandbox')
    # Add a channel dimension for CNNs
    # batched_train_words = np.reshape(batched_train_words, np.shape(batched_train_words) + (1,))
    # batched_test_words = np.reshape(batched_test_words, np.shape(batched_test_words) + (1,))

    model.save_model_as_file('test-init')

    print("Training network...")
    model.train(input_array=batched_train_words,
                label_array=batched_train_cats)
    test_loss, test_recall, test_precision = model.model.evaluate(batched_test_words, batched_test_cats)

    print("Loss: ", test_loss)
    print("Recall: ", test_recall)
    print("Precision: ", test_precision)
    if not (test_precision + test_recall) == 0:
        f1_measure = 2 * (test_precision*test_recall) / (test_precision + test_recall)
    else:
        f1_measure = 0

    print("F1: ", f1_measure)

    model.save_model_as_file('test-trained')
    return


main()
