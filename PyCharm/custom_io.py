# General modules
import pickle
from gensim.models import KeyedVectors
from nltk.corpus import reuters

# Personal modules
import custom_docs as docs


def save_embedding(emb_dict):
    outfile = open('dictionary.p', 'wb')
    pickle.dump(emb_dict, outfile)
    outfile.close()
    return


def load_embedding():
    infile = open('dictionary.p', 'rb')
    emb_dict = pickle.load(infile)
    infile.close()
    return emb_dict


def load_base_embedding():
    # Load embedding vectors directly from the file
    return KeyedVectors.load_word2vec_format('WordEmbeddings/GoogleNews-vectors-negative300.bin', binary=True)


def load_corpus_docs():
    test_docs = []
    train_docs = []

    i = 0
    for fileid in reuters.fileids():
        i += 1
        if 'test' in fileid:
            # test_docs.append((reuters.words(fileid), reuters.sents(fileid)))
            test_docs.append(docs.TokenizedDoc(reuters.words(fileid),
                                               reuters.sents(fileid),
                                               reuters.categories(fileid)))
        elif 'training' in fileid:
            # train_docs.append((reuters.words(fileid), reuters.words(fileid)))
            train_docs.append(docs.TokenizedDoc(reuters.words(fileid),
                                                reuters.sents(fileid),
                                                reuters.categories(fileid)))
        else:
            print("Document not recognized as part of training-set or test-set while extracting the Reuters Corpus")

    # print("Number of documents in this version of the Reuters-Corpus: ", i)
    return train_docs, test_docs


def load_corpus_categories():
    return reuters.categories()

