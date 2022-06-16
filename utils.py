import pickle
import time
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import *

sns.set(font_scale=1.2)
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

    X = X.to(device)
    Y = Y.to(device)

    return X, Y


def elapsed_time(start, end):
    '''
    Helper function to compute elapsed time
    '''
    secs = end - start
    mins = int(secs / 60)
    hrs = int(mins / 60)
    return hrs, mins % 60, int(secs % 60)


class BNN_Dataset(Dataset):
    '''
    Dataset class for creating iterable dataloader
    '''
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
    '''
    Visualise and compare the prediction of a neuronal firing pattern in a colour map.
    :param y:                   true label
    :param y_hat:               prediction
    :param reshape:             tuple (w, h), how the colour map is shown. By default show in a square
    '''
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
    plt.axis('off')
    plt.title('True firing pattern')

    plt.subplot(122)
    plt.imshow(y_hat_r)
    plt.axis('off')
    plt.title('Predicted firing pattern')

    plt.show()
    

def train(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss.
    '''
    start = time.time()

    eval_losses, train_losses = [], []

    init_eval_loss = eval(model, test_loader, criterion)
    if verbose:
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

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
    return train_losses, eval_losses


def eval(model, test_loader, criterion):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()

    with torch.no_grad():

        eval_loss = 0

        for _, (X, Y) in enumerate(iter(test_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    return eval_loss


def train_rnn(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss.
    '''
    start = time.time()

    eval_losses, train_losses = [], []

    init_eval_loss = eval_rnn(model, test_loader, criterion)
    if verbose:
        print(f'Initial eval loss: {init_eval_loss}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        h_prev = torch.zeros([model.n_layers, train_loader.batch_size, model.hidden_dim]).to(device)
        c_prev = torch.zeros([model.n_layers, train_loader.batch_size, model.hidden_dim]).to(device)
        rec_prev = (h_prev, c_prev)

        for i, (X, Y) in enumerate(iter(train_loader)): # X: [batch_size, time_step, input_dim]

            for j in range(X.shape[1]):
            
                Y_hat, rec_prev = model(X[:,j,:], rec_prev)
                loss = criterion(Y_hat, Y)[:,j,:]

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            if force_stop and i == 20:
                break

        eval_loss = eval_rnn(model, test_loader, criterion)

        train_losses.append(epoch_loss)
        eval_losses.append(eval_loss)

        if verbose:
            epoch_end = time.time()
            hrs, mins, secs = elapsed_time(start, epoch_end)
            print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
    
    hrs, mins, secs = elapsed_time(start, time.time())

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
    return train_losses, eval_losses


def eval_rnn(model, test_loader, criterion):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()

    with torch.no_grad():

        eval_loss = 0
        
        h_prev = torch.zeros([model.n_layers, test_loader.batch_size, model.hidden_dim]).to(device)
        c_prev = torch.zeros([model.n_layers, test_loader.batch_size, model.hidden_dim]).to(device)
        rec_prev = (h_prev, c_prev)

        for _, (X, Y) in enumerate(iter(test_loader)):
            for j in range(X.shape[1]):
            
                Y_hat, c_prev = model(X[:,j,:], rec_prev)
                loss = criterion(Y_hat, Y[:,j,:])
            
            eval_loss += loss.item()
    
    return eval_loss


def plot_loss_curves(train_losses, eval_losses, loss_func='MSE loss'):
    '''
    Plots the loss history per epoch.
    '''
    n_epochs = len(train_losses)
    
    plt.figure(figsize=(12,6))
    plt.plot(train_losses)
    plt.plot(eval_losses)

    plt.legend(['train', 'eval'])
    
    plt.xlabel('Epochs')
    plt.ylabel(loss_func)

    plt.title(f'Training and evaluation {loss_func} curve over {n_epochs} epochs')
    plt.show()


def param_grid_search(hidden_dims, n_layerss, **kwargs):
    
    _type = kwargs['_type']
    valid_loader = kwargs['valid_loader']
    test_loader = kwargs['valid_loader']
    n_epochs = kwargs['n_epochs']

    final_eval_loss = np.zeros((len(hidden_dims), len(n_layerss)))

    for i, hidden_dim in enumerate(hidden_dims):
        for j, n_layers in enumerate(n_layerss):

            # deep learning model
            if _type == 'FF':
                DNN = FeedForwardDNN(input_dim=16, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=16).to(device)
            elif _type == 'RNN':
                DNN = RecurrentDNN(input_dim=16, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=16).to(device)

            # training parameters
            optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
            criterion = nn.MSELoss()    

            train_losses, eval_losses = train(
                model=DNN,
                train_loader=valid_loader, test_loader=test_loader,
                optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
                verbose=False, force_stop=False)

            print(f'{n_layers} layers, {hidden_dim} hidden units, final eval loss: {eval_losses[-1]}', end='\r')
            final_eval_loss[i,j] = eval_losses[-1]
    
    return final_eval_loss

def repeated_param_grid_search(hidden_dims, n_layerss, n_repeats, **kwargs):

    final_eval_loss_matrix = np.zeros((len(hidden_dims), len(n_layerss)))
    start = time.time()
    
    for i in range(n_repeats):
        
        final_eval_loss = param_grid_search(hidden_dims, n_layerss, **kwargs)
        final_eval_loss_matrix += final_eval_loss
        
        repeat_end = time.time()
        elapsed_h, elapsed_m, elapsed_s = elapsed_time(start, repeat_end)
        
        r, c = find_argmin_in_matrix(final_eval_loss)
        print(f'{i+1}th repeat: best hidden unit: {hidden_dims[r]}, best layers: {n_layerss[c]}, final eval loss: {final_eval_loss[r,c]}, time elapsed: {elapsed_h} h {elapsed_m} m {elapsed_s} s')
    
    return final_eval_loss_matrix / n_repeats


def find_argmin_in_matrix(mat):
    '''
    Find the row and coloumn of the smallest element in a matrix
    '''
    nr, nc = mat.shape
    return int(np.argmin(mat)/nc), np.argmin(mat) - int(np.argmin(mat)/nc) * nc


def plot_3d_scatter(x, y, z, x_label, y_label, z_label, fname=False):
    '''
    Produces 3d scatter plot
    '''
    xyz = np.zeros([len(x)*len(y), 3])
    for i in range(len(x)):
        for j in range(len(y)):
            xyz[i*len(x)+j,:] = np.array([x[i], y[j], z[i,j]])
    
    plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt3d = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=xyz[:,2])
    cbar = plt.colorbar(plt3d)
    cbar.set_label(z_label)
    if fname:
        plt.savefig('./figures/' + fname)
    plt.show()


def plot_bvc_firing_field(bvc, max_d=2500, n=200):
    rads = np.linspace(-np.pi, np.pi, n)
    ds = np.linspace(0, max_d, n)

    rads_mat, ds_mat = np.meshgrid(rads, ds)

    plt.figure(figsize=(4,4))
    plt.axes(projection='polar')

    firing_rates = bvc.obtain_firing_rate(ds_mat, rads_mat)
    plt.scatter(rads_mat, ds_mat, c=firing_rates, s=1, cmap='hsv', alpha=0.75)
    plt.show()