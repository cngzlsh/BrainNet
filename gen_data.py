from approx_bnn import ApproximateBNN

import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
import os
import pickle

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(x, y, z, input_dim, output_dim, transfer_function, bias, trainable, state_dict=False):
    """
    Loads an approximate BNN model
    :param x:                   number of hidden units of the approximate biological network
    :param y:                   connectivity of network
    :param z:                   number of layers of the biological network 
    :param input_dim:           input dimension
    :param output_dim:          output dimension
    :param transfer_function:   transfer function used in approximate bnn
    :param bias:                whether to use bias in forward layers
    :param trainable:           whether the approximate network is trainable
    :param state_dict:          file path to saved model weights, if applicable
    :return:
    approx_bnn:                 loaded MLP
    """
    approx_bnn = ApproximateBNN(x, y, z, input_dim, output_dim, transfer_function, bias=bias, trainable=trainable).to(device)

    if state_dict:
        approx_bnn.load_state_dict(torch.load(state_dict))

    return approx_bnn


def generate_firing_pattern(model, input_dim, num_input, firing_prob):
    """
    Generates a toy dateset of firing pattern of approximate BNN
    :param model:               MLP, approximate biological network
    :param input_dim:           input dimension
    :param output_dim:          output dimension
    :param num_inputs:          number of datapoints to generate
    :param firing_prob:         firing probability of presynaptic neurons,
                                scalar or numpy array (if different for each input neuron)
    :return:
    X:                          list of input neuron firing patterns (=1 if fires, =0 if not)
    Y:                          list of output firing patterns
    """
    
    if isinstance(firing_prob, float):
        firing_prob = (torch.ones(input_dim) * firing_prob).to(device)
    else:
        firing_prob = torch.Tensor([firing_prob]).to(device)
    
    assert firing_prob.shape == torch.Size([input_dim]), 'Number of firing rates is not equal to the number of input neurons'
    assert torch.min(firing_prob) > 0, 'Firing probability must be between 0 and 1'
    assert torch.max(firing_prob) < 1, 'Firing probability must be between 0 and 1'

    X = dist.Bernoulli(probs=firing_prob).sample(sample_shape=torch.Size([num_input]))
    Y = model(X)

    return X, Y

def save_data(X, Y, path, filename):

    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + filename, 'wb') as f:
        pickle.dump((X, Y), f)
    
    f.close()


if __name__ == '__main__':
    x = 64               # number of hidden units in each layer
    y = 0.8             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    transfer_function = nn.ReLU() 
    print(f'Network connectivity: {x * y * z}')

    input_dim = 16
    output_dim = 16
    num_input = 1000
    firing_prob = 0.5

    approx_bnn = load_model(x, y, z, input_dim, output_dim, transfer_function, bias, trainable, state_dict=False)

    X, Y = generate_firing_pattern(model=approx_bnn, input_dim=input_dim, num_input=num_input, firing_prob=firing_prob)

    save_data(X, Y, './data/', 'test.pkl')