import pickle

def load_data(path, filename):
    with open(path + filename, 'rb') as f:
        X, Y = pickle.load(f)
    
    f.close()
    
    return X, Y