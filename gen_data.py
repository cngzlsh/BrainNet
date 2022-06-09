from tkinter import Y
from sklearn import gaussian_process
from approx_bnns import *
from bvc import *
from utils import *

import torch
import torch.nn as nn
import torch.distributions as dist

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_BNN(x, y, z, input_dim, output_dim, transfer_function, bias, trainable, state_dict=False):
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
    approx_bnn = FeedForwardApproximateBNN(x, y, z, input_dim, output_dim, transfer_function, bias=bias, trainable=trainable).to(device)
    approx_bnn.to(device)

    if state_dict:
        approx_bnn.load_state_dict(torch.load(state_dict))

    return approx_bnn


def generate_binary_firing_pattern(BNN, input_dim, num_input, firing_prob, gaussian_noise=False):
    """
    Generates a toy dateset of firing pattern of approximate BNN
    :param model:               MLP, approximate biological network
    :param input_dim:           input dimension
    :param output_dim:          output dimension
    :param num_inputs:          number of datapoints to generate
    :param firing_prob:         firing probability of presynaptic neurons,
                                scalar or numpy array (if different for each input neuron)
    :param gaussian_noise:      (mean, std), whether to add noise to outputs. Default False
    :return:
    X:                          list of input neuron firing patterns (=1 if fires, =0 if not)
    Y:                          list of output firing patterns
    """
    
    if isinstance(firing_prob, float):
        firing_prob = (torch.ones(input_dim) * firing_prob).to(device)
    else:
        firing_prob = firing_prob.to(device)
    
    assert firing_prob.shape == torch.Size([input_dim]), 'Number of firing rates is not equal to the number of input neurons'
    assert torch.min(firing_prob) > 0, 'Firing probability must be between 0 and 1'
    assert torch.max(firing_prob) < 1, 'Firing probability must be between 0 and 1'

    X = 2 * dist.Bernoulli(probs=firing_prob).sample(sample_shape=torch.Size([num_input])) - 1
    Y = BNN(X)

    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]

    return X.cpu(), Y.cpu()


def generate_bvc_network_firing_pattern(n_data_points, n_cells, preferred_distances, preferred_orientations, sigma_rads, sigma_angs):
    
    BVCs = [BVC(r= preferred_distances[i], theta=preferred_orientations[i], sigma_rad=sigma_rads[i], sigma_ang=sigma_angs[i], scaling_factor=1) for i in range(n_cells)]
    model = BVCNetwork(BVCs=BVCs, coeff=1, threshold=0, non_linearity=nn.ReLU())

    ds = dist.uniform.Uniform(low=0, high=10).sample(sample_shape=torch.Size([n_data_points]))
    phis = dist.uniform.Uniform(low=-torch.pi, high=torch.pi).sample(sample_shape=torch.Size([n_data_points]))

    X = torch.stack((ds, phis), dim=-1)
    Y = model.obtain_firing_rate(ds, phis)[:, None]
    return X.cpu(), Y.cpu()


if __name__ == '__main__':
    x = 256             # number of hidden units in each layer
    y = 0.5             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    transfer_function = nn.ReLU() 
    print(f'Network connectivity: {x * y * z}')

    input_dim = 16
    output_dim = 16
    num_train_input = 10000
    num_test_input = 1000
    firing_prob = 0.5
    gaussian_noise = (torch.Tensor([0]), torch.Tensor([0.001]))

    approx_bnn = load_BNN(x, y, z, input_dim, output_dim, transfer_function, bias, trainable, state_dict=False)

    X_train, Y_train = generate_binary_firing_pattern(BNN=approx_bnn, input_dim=input_dim, num_input=num_train_input, firing_prob=firing_prob, gaussian_noise=False)
    save_data(X_train, Y_train, './data/', f'train_abnn_{input_dim}_{x}_{y}_{z}_{output_dim}_{firing_prob}_{num_train_input}.pkl')

    X_test, Y_test = generate_binary_firing_pattern(BNN=approx_bnn, input_dim=input_dim, num_input=num_test_input, firing_prob=firing_prob, gaussian_noise=False)
    save_data(X_test, Y_test, './data/', f'test_abnn_{input_dim}_{x}_{y}_{z}_{output_dim}_{firing_prob}_{num_test_input}.pkl')


    # n_cells = 8             # number of BVCs to simulate
    # num_train_input = 10000
    # num_test_input = 1000

    # # BVC preferred distances ~ Uniform(0, 10)
    # preferred_distances = dist.uniform.Uniform(low=0, high=10).sample(torch.Size([n_cells]))
    # # BVC preferred angles ~ Uniform(-pi, pi)
    # preferred_orientations = dist.uniform.Uniform(low=-torch.pi, high=torch.pi).sample(torch.Size([n_cells]))
    # sigma_rads = torch.ones(n_cells)
    # sigma_angs = torch.ones(n_cells)

    # X_train, Y_train = generate_bvc_network_firing_pattern(n_data_points=num_train_input, n_cells=n_cells, preferred_distances=preferred_distances, preferred_orientations=preferred_orientations, sigma_rads=sigma_rads, sigma_angs=sigma_angs)
    # save_data(X_train, Y_train, './data/', f'train_bvc.pkl')
    # X_test, Y_test = generate_bvc_network_firing_pattern(n_data_points=num_test_input, n_cells=n_cells, preferred_distances=preferred_distances, preferred_orientations=preferred_orientations, sigma_rads=sigma_rads, sigma_angs=sigma_angs)
    # save_data(X_test, Y_test, './data/', f'test_bvc.pkl')