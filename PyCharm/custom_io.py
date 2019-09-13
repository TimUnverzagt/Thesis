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


def corpus_stats(docs):
    sents_in_doc = []
    words_in_doc = []
    words_in_sent = []
    for doc in docs:
        words_in_doc.append(len(doc[0]))
        sents_in_doc.append(len(doc[1]))
        for sent in doc[1]:
            words_in_sent.append(len(sent))

    print("")
    print('-' * 50)
    print('-' * 50)
    print("Minimal number of 'words' in a text: ", min(words_in_doc))
    print("Average number of 'words' in a text: ", sum(words_in_doc) / len(docs))
    print("Maximal number of 'words' in a text: ", max(words_in_doc))
    print("")
    print("Minimal number of 'words' in a sentence: ", min(words_in_sent))
    print("Average number of 'words' in a sentence: ", sum(words_in_sent) / len(words_in_sent))
    print("Maximal number of 'words' in a sentence: ", max(words_in_sent))
    print("")
    print("Minimal number of sentences in a text: ", min(sents_in_doc))
    print("Average number of sentences in a text: ", sum(sents_in_doc) / len(docs))
    print("Maximal number of sentences in a text: ", max(sents_in_doc))
    print("")
    print("Total number of words in the corpus: ", sum(words_in_doc))
    print('-' * 50)
    print('-' * 50)
    print("")
    return


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

