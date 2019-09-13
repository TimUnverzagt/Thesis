# General modules
import pickle
from gensim.models import KeyedVectors
from nltk.corpus import reuters


def save_embedding(embedding_dict):
    outfile = open('dictionary.p', 'wb')
    pickle.dump(embedding_dict, outfile)
    outfile.close()
    return


def load_embedding():
    infile = open('dictionary.p', 'rb')
    embedding_dict = pickle.load(infile)
    infile.close()
    return embedding_dict


def load_base_embedding():
    # Load embedding vectors directly from the file
    return KeyedVectors.load_word2vec_format('WordEmbeddings/GoogleNews-vectors-negative300.bin', binary=True)


def load_corpus():
    test_docs = []
    train_docs = []

    for fileid in reuters.fileids():
        if 'test' in fileid:
            test_docs.append((reuters.words(fileid), reuters.sents(fileid)))
        elif 'training' in fileid:
            train_docs.append((reuters.words(fileid), reuters.words(fileid)))
        else:
            print("Document not recognized as part of training-set or test-set")

    return train_docs, test_docs

