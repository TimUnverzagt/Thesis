import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import reuters
from nltk.probability import FreqDist


def load_embedding():
    # Load embedding vectors directly from the file
    return KeyedVectors.load_word2vec_format('WordEmbeddings/GoogleNews-vectors-negative300.bin', binary=True)


def embed_word(model, word):
    if word in model:
        return model[word]
    else:
        # print("Word is not contained in the Language Model and will be randomly initialized.")
        new_representation = np.random.rand(300)
        # model.add(word, new_representation)
        return new_representation


def embed_doc(model, doc):
    embedded_words = []
    for word in doc[0]:
        embedded_words.append(embed_word(model, word))
    return np.array(embedded_words)


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


def embedding_stats(model, docs):
    words = []
    words_in_model = []
    words_out_of_model = []
    for doc in docs:
        for word in doc[0]:
            words.append(word)
            if word in model:
                words_in_model.append(word)
            else:
                words_out_of_model.append(word)

    no_in = len(words_in_model)
    no_out = len(words_out_of_model)
    unique_in = len(set(words_in_model))
    unique_out = len(set(words_out_of_model))

    fdist_out = FreqDist(word.lower() for word in words_out_of_model)

    print("")
    print('-'*50)
    print('-'*50)
    print("Amount of words in the model: ", no_in)
    print("Unique words in the model: ", unique_in)
    print("")
    print("Amount of words outside the model: ", no_out)
    print("Unique words outside the model: ", unique_out)
    print("")
    print("Percentage of words outside the model: ", round(100 * no_out / len(words), 1))
    print("Percentage of unique words outside the model: ", round(100 * unique_out / len(set(words)), 1))
    print("")
    print("Most common words outside the model:")
    print(fdist_out.most_common(10))
    print('-'*50)
    print('-'*50)
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
    # docs are tupels (word_tokenizing, sentence_tokenizing)
    [train_docs, test_docs] = load_corpus()

    model = load_embedding()
    # embedding_stats(model, train_doc_words)
    print(np.shape(embed_doc(model, train_docs[0])))

    # corpus_stats(train_docs)
    # corpus_stats(test_docs)


main()

