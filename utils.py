import pickle
import torch
import torch.nn as nn
import time
import os

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

    return X, Y


def elapsed_time(start, end):
    '''
    Helper function to compute elapsed time
    '''
    secs = end - start
    mins = int(secs / 60)
    hrs = int(mins / 60)
    return hrs, mins % 60, secs % 60


def create_mini_batches(X, Y, batch_size):
    '''
    Creates mini batches from data
    :param X:                   training data, shape (N_data, input_dim)
    :param Y:                   training label, shape (N_data, output_dim)
    :param batch_size:          size of mini_batches, 1 <= batch_size <= N_data
    :return:
    X_batch:                    mini-batch training data, shape (n_batches, batch_size, input_dim)
    Y_batch:                    mini-batch training label, shape (n_batches, batch_size, output_dim)
    '''
    
    X_batch = torch.zeros_like



def train(model, train_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False):
    '''
    Main training function. Iterates through train_loader
    '''
    model.train()
    start = time.time()

    for epoch in range(num_epochs):
        
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
        
        if verbose:
            epoch_end = time.time()
            hrs, mins, secs = elapsed_time(start, epoch_end)
            print(f'Epoch {epoch+1} completed with training loss {int(epoch_loss)}. Time elapsed: {hrs} h {mins} m {secs} s.')
        
        if force_stop and i == 20:
            break
    
    end = time.time()
    hrs, mins, secs = elapsed_time(start, time.time())

    print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {hrs} h {mins} m {secs} s.')


def eval(model, eval_loader, criterion):
    '''
    Evaluation function.
    '''
    model.eval()

    with torch.no_grad():

        eval_loss = 0

        for i, (X, Y) in iter(eval_loader):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    return eval_loss
