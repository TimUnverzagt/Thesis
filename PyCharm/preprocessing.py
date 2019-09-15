# General modules
import numpy as np
from nltk.probability import FreqDist

# Personal modules
import custom_io as io
import custom_docs as docs


def prepare_embedding(tok_docs):
    # TODO: I could try out casting every word in the docs to .lower() before lookup in BKV
    base_keyed_vectors = io.load_base_embedding()
    padded_dict = {}
    for word in base_keyed_vectors.vocab:
        padded_dict[word] = base_keyed_vectors[word]
    for doc in tok_docs:
        for word in doc[0]:
            if word not in padded_dict:
                # Randomly initiate key if the word is unknown
                # TODO: New keys might be initialized slightly less naive even without training
                padded_dict[word] = np.random.rand(300)

    return padded_dict


def embed_word(emb_dict, word):
    try:
        return emb_dict[word]
    except KeyError:
        print("The word you are trying is not in the embedding dictionary!")
        print("Check if the embedding was correctly prepared/loaded!")


def embed_categories(doc_cats):
    possible_cats = io.load_corpus_categories()
    emb_cats = np.zeros(len(possible_cats))
    for index, cat in enumerate(possible_cats):
        if cat in doc_cats:
            emb_cats[index] = 1
    return emb_cats


def embed_doc(emb_dict, tok_doc):
    emb_words = []
    for word in tok_doc[0]:
        emb_words.append(embed_word(emb_dict, word))
    word_embedding = np.array(emb_words)

    sent_embeddings = []
    for sent in tok_doc[1]:
        emb_words_in_sent = []
        for word in sent:
            emb_words_in_sent.append(embed_word(emb_dict, word))
        sentence_embedding = np.array(emb_words_in_sent)
        sent_embeddings.append(sentence_embedding)

    cats_embedding = embed_categories(tok_doc[2])

    return docs.EmbeddedDoc(word_embedding, sent_embeddings, cats_embedding)


def embed_docs(emb_dict, tok_docs):
    emb_docs = []
    for tok_doc in tok_docs:
        emb_docs.append(embed_doc(emb_dict, tok_doc))

    return emb_docs


def batch_docs(emb_docs, target_doc_len):
    # TODO: Implement
    no_of_docs = len(emb_docs)
    no_of_cats = len(io.load_corpus_categories())
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

