# General modules
import numpy as np
from nltk.probability import FreqDist

# Personal modules
import custom_io as io
import custom_docs as docs


def prepare_embedding(tokenized_docs):
    # TODO: I could try out casting every word in the docs to .lower() before lookup in BKV
    base_keyed_vectors = io.load_base_embedding()
    padded_dict = {}
    for word in base_keyed_vectors.vocab:
        padded_dict[word] = base_keyed_vectors[word]
    for doc in tokenized_docs:
        for word in doc[0]:
            if word not in padded_dict:
                # Randomly initiate key if the word is unknown
                # TODO: New keys might be initialized slightly less naive even without training
                padded_dict[word] = np.random.rand(300)

    return padded_dict


def embed_word(embedding_dict, word):
    try:
        return embedding_dict[word]
    except KeyError:
        print("The word you are trying is not in the embedding dictionary!")
        print("Check if the embedding was correctly prepared/loaded!")


def embed_categories(doc_categories):
    possible_categories = io.load_corpus_categories()
    embedded_categories = np.zeros(len(possible_categories))
    for index, category in enumerate(possible_categories):
        if category in doc_categories:
            embedded_categories[index] = 1
    return embedded_categories


def embed_doc(embedding_dict, tokenized_doc):
    embedded_words = []
    for word in tokenized_doc[0]:
        embedded_words.append(embed_word(embedding_dict, word))
    word_embedding = np.array(embedded_words)

    sent_embeddings = []
    for sent in tokenized_doc[1]:
        embedded_words_in_sent = []
        for word in sent:
            embedded_words_in_sent.append(embed_word(embedding_dict, word))
        sentence_embedding = np.array(embedded_words_in_sent)
        sent_embeddings.append(sentence_embedding)

    categories_embedding = embed_categories(tokenized_doc[2])

    return docs.EmbeddedDoc(word_embedding, sent_embeddings, categories_embedding)


def embed_docs(embedding_dict, tokenized_docs):
    embedded_docs = []
    for tokenized_doc in tokenized_docs:
        embedded_docs.append(embed_doc(embedding_dict, tokenized_doc))

    return embedded_docs


def embedding_stats(embedding_dict, tokenized_docs):
    words = []
    words_in_model = []
    words_out_of_model = []
    for doc in tokenized_docs:
        for word in doc[0]:
            words.append(word)
            if word in embedding_dict:
                words_in_model.append(word)
            else:
                words_out_of_model.append(word)

    no_in = len(words_in_model)
    no_out = len(words_out_of_model)
    unique_in = len(set(words_in_model))
    unique_out = len(set(words_out_of_model))

    freq_dist_out = FreqDist(word.lower() for word in words_out_of_model)

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
    print(freq_dist_out.most_common(10))
    print('-'*50)
    print('-'*50)
    print("")
    return


def corpus_stats(tokenized_docs):
    sents_in_doc = []
    words_in_doc = []
    words_in_sent = []
    for doc in tokenized_docs:
        words_in_doc.append(len(doc[0]))
        sents_in_doc.append(len(doc[1]))
        for sent in doc[1]:
            words_in_sent.append(len(sent))

    print("")
    print('-' * 50)
    print('-' * 50)
    print("Minimal number of 'words' in a text: ", min(words_in_doc))
    print("Average number of 'words' in a text: ", sum(words_in_doc) / len(tokenized_docs))
    print("Maximal number of 'words' in a text: ", max(words_in_doc))
    print("")
    print("Minimal number of 'words' in a sentence: ", min(words_in_sent))
    print("Average number of 'words' in a sentence: ", sum(words_in_sent) / len(words_in_sent))
    print("Maximal number of 'words' in a sentence: ", max(words_in_sent))
    print("")
    print("Minimal number of sentences in a text: ", min(sents_in_doc))
    print("Average number of sentences in a text: ", sum(sents_in_doc) / len(tokenized_docs))
    print("Maximal number of sentences in a text: ", max(sents_in_doc))
    print("")
    print("Total number of words in the corpus: ", sum(words_in_doc))
    print('-' * 50)
    print('-' * 50)
    print("")
    return

