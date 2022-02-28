from approx_bnn import ApproximateBNN

import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist

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


def generate_firing_pattern(model, input_dim, num_input, firing_rates):
    """
    Generates a toy dateset of firing pattern of approximate BNN
    :param model:               MLP, approximate biological network
    :param input_dim:           input dimension
    :param output_dim:          output dimension
    :param num_inputs:          number of datapoints to generate
    :param firing_rate:         mean firing rate of each input neuron between 0 and 10, 
                                scalar or numpy array (if different for each input neuron)
    :return:
    X:                          list of input rates
    Y:                          list of output firing patterns
    """
    if isinstance(firing_rates, int or float):
        firing_rates = (torch.ones(input_dim) * firing_rates).to(device)
    else:
        firing_rates = torch.Tensor(firing_rates).to(device)
    
    assert firing_rates.shape == torch.Size([input_dim]), 'Number of firing rates is not equal to the number of input neurons'
    assert torch.min(firing_rates) > 0, 'Firing rate must be positive'
    assert torch.max(firing_rates) < 10, 'Firing rate must be smaller than 10'

    X = dist.Poisson(rate = firing_rates).sample(sample_shape=torch.Size([num_input]))

    Y = model(X)
    print(Y, Y.shape)
    return X, Y

if __name__ == '__main__':
    x = 64               # number of hidden units in each layer
    y = 0.8             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    transfer_function = nn.ReLU() 
    print(f'Network connectivity: {x * y * z}')

    input_dim = 4
    output_dim = 2
    num_input = 1000
    firing_rates = 6

    approx_bnn = load_model(x, y, z, input_dim, output_dim, transfer_function, bias, trainable, state_dict=False)

    X, Y = generate_firing_pattern(model=approx_bnn, input_dim=input_dim, num_input=num_input, firing_rates=firing_rates)