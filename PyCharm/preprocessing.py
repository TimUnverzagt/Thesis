import numpy as np
import pickle
from gensim.models import KeyedVectors
from nltk.probability import FreqDist


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


def prepare_embedding(tokenized_docs):
    # TODO: I could try out casting every word in the docs to .lower() before lookup in BKV
    base_keyed_vectors = load_base_embedding()
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


def embed_doc(embedding_dict, tokenized_doc):
    embedded_words = []
    for word in tokenized_doc[0]:
        embedded_words.append(embed_word(embedding_dict, word))
    return np.array(embedded_words)


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

