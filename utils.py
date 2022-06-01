import pickle
import time
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

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

def visualise_prediction(y, y_hat, reshape='square'):

    if reshape == 'square':
        dim = len(y)
        w, l = int(np.sqrt(dim)), int(np.sqrt(dim))
    else:
        w, l = reshape
    
    try:
        y_r = y.reshape((w,l)).cpu()
        y_hat_r = y_hat.reshape(w,l).cpu().detach().numpy()
    except:
        raise ValueError('Reshape dimension mismatch')

    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.imshow(y_r)
    plt.title('True firing pattern')

    plt.subplot(122)
    plt.imshow(y_hat_r)
    plt.title('Predicted firing pattern')

    plt.show()
    

def train(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss
    '''
    
    start = time.time()

    eval_losses, train_losses = [], []

    init_eval_loss = eval(model, test_loader, criterion)
    print(f'Initial eval loss: {init_eval_loss}')

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

        train_losses.append(epoch_loss)
        eval_losses.append(eval_loss)

        if verbose:
            epoch_end = time.time()
            hrs, mins, secs = elapsed_time(start, epoch_end)
            print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
    
    hrs, mins, secs = elapsed_time(start, time.time())

    print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
    return train_losses, eval_losses


def eval(model, test_loader, criterion):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()

    with torch.no_grad():

        eval_loss = 0

        for i, (X, Y) in enumerate(iter(test_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    return eval_loss

def plot_loss_curve(train_losses, eval_losses, loss_func='MSE loss'):
    n_epochs = len(train_losses)
    
    plt.figure(figsize=(12,6))
    plt.plot(train_losses)
    plt.plot(eval_losses)

    plt.legend(['train', 'eval'])
    
    plt.xlabel('Epochs')
    plt.ylabel(loss_func)

    plt.title(f'Training and evaluation {loss_func} curve over {n_epochs} epochs')
    plt.show()
