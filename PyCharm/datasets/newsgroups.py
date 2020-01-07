# General modules
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups

# personal modules
from datasets import preprocessing as pre


def quantify_datapoints(target_doc_len):
    vocabulary = get_vocabulary()

    print("Encoding the documents...")
    training_datapoints = translate_wrapper(old_wrapper=fetch_20newsgroups(subset='train'),
                                            target_doc_len=target_doc_len,
                                            vocabulary=vocabulary)
    test_datapoints = translate_wrapper(old_wrapper=fetch_20newsgroups(subset='test'),
                                        target_doc_len=target_doc_len,
                                        vocabulary=vocabulary)
    return {'train': training_datapoints, 'test': test_datapoints}


def get_vocabulary():
    vocabulary = {}
    with open('datasets/newsgroups_vocabulary') as txt_file:
        for idx, line in enumerate(txt_file):
            vocabulary[line.rstrip('\n')] = idx
    return vocabulary


def translate_wrapper(old_wrapper, target_doc_len, vocabulary):
    emb_data = []
    for idx, text in enumerate(old_wrapper.data):
        if ((idx % 1000) == 0) & (idx != 0):
            print(str(idx) + " documents have been encoded.")
        tok_text = pre.tokenize(text, lower=True, head_stripper="\n\n")
        emb_text = pre.embed(tok_text=tok_text, vocabulary=vocabulary)
        emb_data.append(pre.unify_length(tok_doc=emb_text,
                                         target_length=target_doc_len,
                                         padding='zero'))
    train_array = np.zeros(shape=(len(old_wrapper.data), target_doc_len), dtype=np.int)
    for i, doc in enumerate(emb_data):
        for j in range(len(doc)):
            train_array[i][j] = doc[j]
    datapoints = (train_array, old_wrapper.target)
    print("-" * 10)
    return datapoints
