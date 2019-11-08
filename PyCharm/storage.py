# General modules
import numpy as np
import pickle


def save_experimental_history(history, name='experiment'):
    # TODO: Check for collision by name
    if name == 'experiment':
        print('Saving an experimental history without a chosen name causes a fallback to the default name.')
    outfile = open('Histories/' + name + '_history.p', 'wb')
    pickle.dump(history, outfile)
    outfile.close()
    return


def load_experimental_history(name):
    # TODO: Check whether the file could be confused due to an earlier name collision
    infile = open('Histories' + name + '_history.p', 'rb')
    emb_dict = pickle.load(infile)
    infile.close()
    return emb_dict

