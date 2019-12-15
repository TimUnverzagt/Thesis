# General modules
import numpy as np
from sklearn.datasets import fetch_20newsgroups

# personal modules
from datasets import preprocessing as pre


def quantify_datapoints():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    vocabulary = get_vocabulary()

    print("Embedding the documents...")
    emb_train_data = []
    for idx, text in enumerate(newsgroups_train.data):
        if ((idx % 1000) == 0) & (idx != 0):
            print(str(idx) + " training documents have been embedded.")
        tok_text = pre.tokenize(text, lower=True, head_stripper="\n\n")
        emb_train_data.append(pre.embed(tok_text=tok_text, vocabulary=vocabulary))
    trainining_datapoints = (emb_train_data, newsgroups_train.target)

    emb_test_data = []
    for idx, text in enumerate(newsgroups_test.data):
        if ((idx % 1000) == 0) & (idx != 0):
            print(str(idx) + " testing documents have been embedded.")
        tok_text = pre.tokenize(text, lower=True, head_stripper="\n\n")
        emb_test_data.append(pre.embed(tok_text=tok_text, vocabulary=vocabulary))
    test_datapoints = (emb_test_data, newsgroups_test.target)

    return {'train': trainining_datapoints, 'test': test_datapoints}


def get_vocabulary():
    vocabulary = {}
    with open('datasets/newsgroups_vocabulary') as txt_file:
        for idx, line in enumerate(txt_file):
            vocabulary[line.rstrip('\n')] = idx
    return vocabulary

