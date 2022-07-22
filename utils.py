import pickle
import time
import os

import torch
from torch.utils.data import Dataset
import torch.distributions as dist
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


def save_non_linearities(_dict, filepath):
    '''
    Saves the non-linearities of a biological neural network
    '''
    with open(filepath, 'wb') as f:
        pickle.dump(_dict, f)

    f.close()


def load_non_linearities(filepath):
    with open(filepath, 'rb') as f:
        _dict = pickle.load(f)
    f.close()
    return _dict

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


def find_argmin_in_matrix(mat):
    '''
    Find the row and coloumn of the smallest element in a matrix
    '''
    nr, nc = mat.shape
    return int(np.argmin(mat)/nc), np.argmin(mat) - int(np.argmin(mat)/nc) * nc


def plot_3d_scatter(x, y, z, x_label, y_label, z_label, colorbar=True, fname=False, figsize=(12,10)):
    '''
    Produces 3d scatter plot
    '''
    xyz = np.zeros([len(x)*len(y), 3])
    for i in range(len(x)):
        for j in range(len(y)):
            xyz[i*len(x)+j,:] = np.array([x[i], y[j], z[i,j]])
    
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt3d = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=xyz[:,2])
    if colorbar:
        cbar = plt.colorbar(plt3d)
        cbar.set_label(z_label)
    if fname:
        plt.savefig('./figures/' + fname)
    plt.show()


def plot_bvc_firing_field(bvcs, max_d='auto', n=200):
    '''
    Plots firing field of (multiple) BVCs
    '''
    if not isinstance(bvcs, list):
        bvcs = [bvcs]
    n_bvcs = len(bvcs)
    
    if max_d =='auto':
        max_d = int(max([i.d for i in bvcs]) * 1.5)
    rads = torch.linspace(0, 2*torch.pi, n)
    ds = torch.linspace(0, max_d, n)

    rads_mat, ds_mat = torch.meshgrid(rads, ds)

    plt.figure(figsize=(4*n_bvcs, 4))
    for i in range(n_bvcs):
        ax = plt.subplot(1, n_bvcs,i+1, projection='polar')
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N', offset=0)
        firing_rates = bvcs[i].compute_BVC_firing_single_segment(ds_mat, rads_mat)
        ax.scatter(rads_mat, ds_mat, c=firing_rates, s=1, cmap='hsv', alpha=0.75)
    plt.show()


def generate_locations(length, width, n):
    '''
    Generates multiple random location
    '''
    xs = dist.Uniform(0, length).sample(sample_shape=torch.Size([n]))
    ys = dist.Uniform(0, width).sample(sample_shape=torch.Size([n]))

    return xs, ys

def calc_entropy(x:torch.Tensor):
    '''
    Computes the (Shannon) entropy of a tensor based on its empirical distribution
    '''
    x = x.flatten()
    freq = x.unique(return_counts=True)[1]
    probs = freq/torch.sum(freq)
    return -torch.multiply(probs, torch.log(probs)).sum()