from __future__ import absolute_import, division, print_function, unicode_literals

# General modules
import numpy as np
from pathlib import Path

# Personal modules
import preprocessing as prep
import custom_io as io
from network import CustomNetworkHandler as Network

# tf.debugging.set_log_device_placement(True)


def main():
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tok_train_docs, tok_test_docs] = io.load_corpus_docs()

    if Path("dictionary.p").exists():
        print("Loading previous model...")
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
    model = Network(target_doc_len=30)

    print("Training network...")
    model.train(input_array=batched_train_words,
                label_array=batched_train_cats)

    test_loss, test_acc = model.model.evaluate(batched_test_words, batched_test_cats)

    print("Loss: ", test_loss)
    print("Accuracy: ", test_acc)

    return


main()

