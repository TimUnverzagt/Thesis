# General modules
import pickle


def save_experimental_history(history, path, name='experiment'):
    # TODO: Check for collision by name
    if name == 'experiment':
        print('Saving an experimental history without a chosen name causes a fallback to the default name.')
    outfile = open(path + '/' + name + '_history.p', 'wb')
    pickle.dump(history, outfile)
    outfile.close()
    return


def load_experimental_history(path, name):
    # TODO: Check whether the file could be confused due to an earlier name collision
    infile = open(path + '/' + name + '_history.p', 'rb')
    emb_dict = pickle.load(infile)
    infile.close()
    return emb_dict

