import numpy as np
import preprocessing as pre
from nltk.corpus import reuters


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


def main():
    # tokenized_docs are tupels (word_tokenizing, sentence_tokenizing)
    [tokenized_train_docs, tokenized_test_docs] = load_corpus()

    ## Use this code to setup your embedding for the first time
    # embedding_dict = pre.prepare_embedding(tokenized_train_docs + tokenized_test_docs)
    # pre.save_embedding(embedding_dict)

    ## If you already have an embedding load it with this function
    # embedding_dict = pre.load_embedding()
    return


main()

