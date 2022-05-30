import pickle
import torch
import torch.nn as nn
import time
import os

from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_data(X, Y, path, filename):
    '''
    Saves synthetic neuron firing data to pickle file
    '''
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + filename, 'wb') as f:
        pickle.dump((X, Y), f)
    
    f.close()


def load_data(path, filename):
    '''
    Loads synthetic neuron firing data from pickle file
    '''
    with open(path + filename, 'rb') as f:
        X, Y = pickle.load(f)
    f.close()

    X.to(device)
    Y.to(device)

    return X, Y


def elapsed_time(start, end):
    '''
    Helper function to compute elapsed time
    '''
    secs = end - start
    mins = int(secs / 60)
    hrs = int(mins / 60)
    return hrs, mins % 60, secs % 60


class BNN_Dataset(Dataset):
    def __init__(self, X, Y):
        self.inputs = X
        self.labels = Y

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx,:]
        label = self.labels[idx,:]
        return input, label


def train(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False):
    '''
    Main training function. Iterates through train_loader
    '''
    
    start = time.time()

    eval_loss = eval(model, test_loader, criterion)
    print(f'Initial eval loss: {eval_loss}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (X, Y) in enumerate(iter(train_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

            if force_stop and i == 20:
                break

        eval_loss = eval(model, test_loader, criterion)

        if verbose:
            epoch_end = time.time()
            hrs, mins, secs = elapsed_time(start, epoch_end)
            print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
    
    hrs, mins, secs = elapsed_time(start, time.time())

    print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')


def eval(model, test_loader, criterion):
    '''
    Evaluation function.
    '''
    model.eval()

    with torch.no_grad():

        eval_loss = 0

        for i, (X, Y) in enumerate(iter(test_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    return eval_loss
