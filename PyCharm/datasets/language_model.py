# General modules
import pickle
from gensim.models import KeyedVectors
import numpy as np

# Personal modules


# TODO: The custom embedding should be saved together in the same folder as the pre-trained embeddings

def save_dictionary(emb_dict, filename='default'):
    # TODO: Check for collision by name
    if filename == 'default':
        print('Saving a word embedding without a chosen name causes a fallback to the default name.')
    outfile = open(filename + '_dictionary.p', 'wb')
    pickle.dump(emb_dict, outfile)
    outfile.close()
    return


def load_dictionary():
    infile = open('dictionary.p', 'rb')
    emb_dict = pickle.load(infile)
    infile.close()
    return emb_dict


def load_google_dictionary():
    # Load embedding vectors directly from the file
    base_keyed_vectors = KeyedVectors.load_word2vec_format('WordEmbeddings/GoogleNews-vectors-negative300.bin',
                                                           binary=True)
    google_dict = {}
    # TODO: I could try out casting every word in the docs to .lower() before lookup in BKV
    for word in base_keyed_vectors.vocab:
        google_dict[word] = base_keyed_vectors[word]
    return google_dict


def extend_dictionary(tok_docs, dictionary):
    for doc in tok_docs:
        for word in doc[0]:
            if word not in dictionary:
                # Randomly initiate key if the word is unknown
                # TODO: New keys might be initialized slightly less naive even without training
                dictionary[word] = np.random.rand(300)

    return dictionary
