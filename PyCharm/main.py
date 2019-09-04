from gensim.models import KeyedVectors
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize

def load_embedding():
    # Load embedding vectors directly from the file
    return KeyedVectors.load_word2vec_format('WordEmbeddings/GoogleNews-vectors-negative300.bin', binary=True)

def main():
    #model = load_embedding()

    test_docs = []
    train_docs = []

    for fileid in reuters.fileids():
        if 'test' in fileid:
            test_docs.append(reuters.words(fileid))
        else:
            train_docs.append(reuters.words(fileid))

    embedded_docs = []
    for doc in test_docs:
        doc_embedding = []
        print(doc)
        for word in doc:
            print(word)
            # doc_embedding.append(model[word])
        embedded_docs.append(doc_embedding)

    print(embedded_docs[0])
    # model = load_embedding()
    ## Access vectors for specific words with a keyed lookup:
    # vector = model['easy']
    ## see the shape of the vector (300,)
    # vector.shape
    ## Processing sentences is not as simple as with Spacy:
    # vectors = [model[x] for x in "This is some text I am processing with Spacy".split(' ')]
    # print(vectors)

main()
