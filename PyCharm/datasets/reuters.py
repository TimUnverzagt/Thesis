# General modules
import numpy as np
from nltk.probability import FreqDist
from nltk.corpus import reuters
from typing import NamedTuple
from typing import List
from pathlib import Path

# Personal modules
from datasets import language_model as lm


# Docs as named tuples
class TokenizedDoc(NamedTuple):
    tokenized_words: List[str]
    tokenized_sents: List[List[str]]
    categories: List[str]


class EmbeddedDoc(NamedTuple):
    embedded_words: np.ndarray
    embedded_sents: List[np.ndarray]
    categories: np.ndarray


def embedding_stats(emb_dict, tok_docs):
    words = []
    words_in_model = []
    words_out_of_model = []
    for doc in tok_docs:
        for word in doc[0]:
            words.append(word)
            if word in emb_dict:
                words_in_model.append(word)
            else:
                words_out_of_model.append(word)

    no_in = len(words_in_model)
    no_out = len(words_out_of_model)
    unique_in = len(set(words_in_model))
    unique_out = len(set(words_out_of_model))

    freq_dist_out = FreqDist(word.lower() for word in words_out_of_model)

    print("")
    print('-' * 50)
    print('-' * 50)
    print("Amount of words in the language model: ", no_in)
    print("Unique words in the language model: ", unique_in)
    print("")
    print("Amount of words outside the language model: ", no_out)
    print("Unique words outside the language model: ", unique_out)
    print("")
    print("Percentage of words outside the language model: ", round(100 * no_out / len(words), 1))
    print("Percentage of unique words outside the language model: ", round(100 * unique_out / len(set(words)), 1))
    print("")
    print("Most common words outside the language model:")
    print(freq_dist_out.most_common(10))
    print('-' * 50)
    print('-' * 50)
    print("")
    return


def corpus_stats(tok_docs):
    sents_in_doc = []
    words_in_doc = []
    words_in_sent = []
    for doc in tok_docs:
        words_in_doc.append(len(doc[0]))
        sents_in_doc.append(len(doc[1]))
        for sent in doc[1]:
            words_in_sent.append(len(sent))

    print("")
    print('-' * 50)
    print('-' * 50)
    print("Minimal number of 'words' in a text: ", min(words_in_doc))
    print("Average number of 'words' in a text: ", sum(words_in_doc) / len(tok_docs))
    print("Maximal number of 'words' in a text: ", max(words_in_doc))
    print("")
    print("Minimal number of 'words' in a sentence: ", min(words_in_sent))
    print("Average number of 'words' in a sentence: ", sum(words_in_sent) / len(words_in_sent))
    print("Maximal number of 'words' in a sentence: ", max(words_in_sent))
    print("")
    print("Minimal number of sentences in a text: ", min(sents_in_doc))
    print("Average number of sentences in a text: ", sum(sents_in_doc) / len(tok_docs))
    print("Maximal number of sentences in a text: ", max(sents_in_doc))
    print("")
    print("Total number of words in the corpus: ", sum(words_in_doc))
    print('-' * 50)
    print('-' * 50)
    print("")
    return


def _load_reuters_docs():
    test_docs = []
    train_docs = []

    i = 0
    for fileid in reuters.fileids():
        i += 1
        if 'test' in fileid:
            # test_docs.append((reuters.words(fileid), reuters.sents(fileid)))
            test_docs.append(TokenizedDoc(reuters.words(fileid),
                                          reuters.sents(fileid),
                                          reuters.categories(fileid)))
        elif 'training' in fileid:
            # train_docs.append((reuters.words(fileid), reuters.words(fileid)))
            train_docs.append(TokenizedDoc(reuters.words(fileid),
                                           reuters.sents(fileid),
                                           reuters.categories(fileid)))
        else:
            print("Document not recognized as part of training-set or test-set while extracting the Reuters Corpus")
    return train_docs, test_docs


def _load_reuters_categories():
    return reuters.categories()


def _embed_word(emb_dict, word):
    try:
        return emb_dict[word]
    except KeyError:
        print("The word you are trying is not in the embedding dictionary!")
        print("Check if the embedding was correctly prepared/loaded!")


def _embed_categories(doc_cats):
    possible_cats = _load_reuters_categories()
    emb_cats = np.zeros(len(possible_cats))
    for index, cat in enumerate(possible_cats):
        if cat in doc_cats:
            emb_cats[index] = 1
    return emb_cats


def _embed_doc(emb_dict, tok_doc):
    emb_words = []
    for word in tok_doc[0]:
        emb_words.append(_embed_word(emb_dict, word))
    word_embedding = np.array(emb_words)

    sent_embeddings = []
    for sent in tok_doc[1]:
        emb_words_in_sent = []
        for word in sent:
            emb_words_in_sent.append(_embed_word(emb_dict, word))
        sentence_embedding = np.array(emb_words_in_sent)
        sent_embeddings.append(sentence_embedding)

    cats_embedding = _embed_categories(tok_doc[2])

    return EmbeddedDoc(word_embedding, sent_embeddings, cats_embedding)


def _embed_docs(emb_dict, tok_docs):
    emb_docs = []
    for tok_doc in tok_docs:
        emb_docs.append(_embed_doc(emb_dict, tok_doc))

    return emb_docs


def _batch_docs(emb_docs, target_doc_len):
    no_of_docs = len(emb_docs)
    no_of_cats = len(_load_reuters_categories())
    bat_words = np.zeros(shape=(no_of_docs, target_doc_len, 300))
    bat_cats = np.zeros(shape=(no_of_docs, no_of_cats))
    i = 0
    for index, doc in enumerate(emb_docs):
        no_words_in_doc = np.shape(doc[0])[0]

        # Gather embedding of words
        if no_words_in_doc >= target_doc_len:
            bat_words[index] = doc[0][0:target_doc_len]
        else:
            i += 1
            # Pad documents that are too short (atm implicit zero-padding)
            bat_words[index][0:no_words_in_doc] = doc[0][:]

        # Gather embedding of categories
        bat_cats[index] = doc[2]

    return bat_words, bat_cats


def quantify_datapoints(target_no_of_features):
    [tok_train_docs, tok_test_docs] = _load_reuters_docs()

    if Path("dictionary.p").exists():
        print("Loading previous language model...")
        emb_dictionary = lm.load_dictionary()
    else:
        print("No previous model found!")
        print("Beginning to construct new model with GoogleNews-Vectors as base...")
        emb_dictionary = lm.extend_dictionary(tok_train_docs + tok_test_docs,
                                              lm.load_google_dictionary())
        lm.save_dictionary(emb_dictionary)

    # Embed the docs before you you feed them to the network
    print("Embedding documents...")
    emb_train_docs = _embed_docs(emb_dictionary, tok_train_docs)
    emb_test_docs = _embed_docs(emb_dictionary, tok_test_docs)
    (batched_train_words, batched_train_cats) = _batch_docs(emb_train_docs, target_doc_len=target_no_of_features)
    (batched_test_words, batched_test_cats) = _batch_docs(emb_test_docs, target_doc_len=target_no_of_features)
    datapoints = {'train': (batched_train_words, batched_train_cats),
                  'test': (batched_test_words, batched_test_cats)}
    return datapoints

