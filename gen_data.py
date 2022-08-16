from approx_bnns import *
from bvc import *
from utils import *

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalise_data(data):
    '''
    Normalises data: subtract mean and divide by std over batch_size dim (dim 0)
    '''
    return torch.nan_to_num((data - torch.mean(data, dim=0))/ torch.std(data, dim=0), nan=0.0, posinf=0.0, neginf=0.0)

def generate_binary_firing_pattern(BNN, input_dim, n_data_points, firing_prob, time_steps=False, gaussian_noise=False):
    """
    Generates a toy dateset of firing pattern of approximate BNN
    :param model:               approximate biological network
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

    if not time_steps:
        X = dist.Bernoulli(probs=firing_prob).sample(sample_shape=torch.Size([n_data_points]))
    elif isinstance(time_steps, int):
        X = dist.Bernoulli(probs=firing_prob).sample(sample_shape=torch.Size([n_data_points, time_steps]))
    else:
        raise ValueError('Temporal must be a time length of integer')

    Y = BNN(X)
    # Y = normalise_data(Y)

    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]

    return X.cpu(), Y.cpu()


def generate_stochastic_firing_pattern(BNN, input_dim, n_data_points, mean_freq, gaussian_noise=False):
    '''
    Poisson neuron firing model.

    :param model:               Mapproximate biological network
    :param input_dim:           input dimension
    :param output_dim:          output dimension
    :param num_inputs:          number of datapoints to generate
    :param avg_Rates:           mean firing rate of a neuron
                                if a scalar is provided, assumed identical for each dimension
    :param gaussian_noise:      (mean, std), whether to add noise to outputs. Default False
    :return:
    X:                          list of input neuron firing patterns (=1 if fires, =0 if not)
    Y:                          list of output firing patterns
    '''
    if isinstance(mean_freq, int) or isinstance(mean_freq, float):
        mean_freq = torch.Tensor(torch.ones(input_dim) * mean_freq).to(device)
    
    X = dist.Poisson(rate=mean_freq).sample(sample_shape=torch.Size([n_data_points]))
    
    Y = BNN(X)

    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]

    Y = normalise_data(Y)

    return X.cpu(), Y.cpu()


def gen_multiple_spike_train_counts(alpha, beta, time_steps=50):
    '''
    Generates multiple trains of spikes using Gamma interval process ISI ~ Gamma(alpha, beta)

    :param alphas:      1D torch.Tensor, concentration of each gamma distribution
    :param betas:       1D torch.Tensor, rates of each gamma distrbution. Same shape as alphas
    :time_steps:        int, number of discretised time steps (bins)

    :return:
    bin_counts:         2D torch Tensor [time_steps, dim] number of spikes for each time bin
    '''
    n_trains = alpha.shape[0]

    ts = dist.Gamma(concentration=alpha, rate=beta).sample(sample_shape=torch.Size([1000]))
    cumu_time = torch.cumsum(ts, dim=0)
    bin_counts = torch.vstack([torch.bincount(cumu_time[:,i].long())[:time_steps] for i in range(n_trains)]).permute(1,0)
        
    return bin_counts.float()


def generate_renewal_process_input_pattern(n_data_points, input_dim, alphas, betas, time_steps=50):
    '''
    Renewal process neuron firing model.
    '''
    if isinstance(alphas, int) or isinstance(alphas, float):
        alphas = torch.Tensor(torch.ones(input_dim) * alphas)
        betas = torch.Tensor(torch.ones(input_dim) * betas)

    X = torch.stack([gen_multiple_spike_train_counts(alpha=alphas, beta=betas, time_steps=time_steps) for _ in range( n_data_points)])

    return X


def generate_output_pattern(BNN, X, gaussian_noise=False):

    X = X.to(device)
    Y = BNN(X)
    Y = normalise_data(Y)
    
    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]

    return X.cpu(), Y.cpu()


def apply_plasticity_and_generate_new_output(sigma, alpha=1, **kwargs):
    '''
    Slightly alter each non-zero weight in the biological neural network, and generate new output patterns
    '''
    
    # unpack arguments
    BNN = kwargs['BNN']
    BNN_params = kwargs['BNN_params']
    X_train = kwargs['X_train']
    X_test = kwargs['X_test']
    verbose = True if kwargs['verbose'] == 2 else False
    
    # initialise BNN
    BNN.load_state_dict(BNN_params[0])
    BNN.load_non_linearities(BNN_params[1])
    BNN.gaussian_plasticity_update(sigma, alpha)
    
    if verbose:
        print('\t Plasticity applied.')
    
    _Y_train = BNN(X_train.to(device))
    _Y_test = BNN(X_test.to(device))

    new_Y_train = normalise_data(_Y_train)
    new_Y_test = normalise_data(_Y_test)
    
    if verbose:
        print('\t New output patterns generated.')
    if torch.max(new_Y_train > 1e30):
        raise ValueError('Normalised to nan')

    return new_Y_train, new_Y_test


if __name__ == '__main__':
    x = 256             # number of hidden units in each layer
    y = 0.5             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    residual_in = [False, False, 1, 2]
    gaussian_noise = (torch.Tensor([0]), torch.Tensor([0.01]))

    input_dim = 16
    output_dim = 16
    num_train = 10000
    num_test = 1000
    num_valid = 1000
    mean_freq = 5

    alphas = torch.ones(input_dim) * 2
    betas = dist.Poisson(rate=mean_freq*2).sample(sample_shape=torch.Size([input_dim]))

    time_steps = 50
    transfer_functions=[nn.ReLU(), nn.ELU(), nn.SiLU(), nn.CELU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(0.1), nn.LeakyReLU(0.2), nn.LeakyReLU(0.3)]


    X_train = generate_renewal_process_input_pattern(num_train, input_dim, alphas, betas, time_steps)
    X_test = generate_renewal_process_input_pattern(num_test, input_dim, alphas, betas, time_steps)
    X_valid = generate_renewal_process_input_pattern(num_valid, input_dim, alphas, betas, time_steps)


    approx_bnn = RecurrentApproximateBNN(
        x=x, y=y, z=z,
        input_dim=input_dim,
        output_dim=output_dim, 
        # residual_in=residual_in,
        recurrent_dim=-1,
        transfer_functions=transfer_functions
        ).to(device)


    # torch.save(approx_bnn.state_dict(), './approx_bnn_params/complex.pt')
    # save_non_linearities(approx_bnn.extract_non_linearities(), './approx_bnn_params/complex_activations.pkl')


    X_train, Y_train = generate_output_pattern(
        BNN=approx_bnn, 
        X=X_train,
        gaussian_noise=False)
    save_data(X_train, Y_train, './data/', f'recurrent_train_nonoise.pkl')

    X_test, Y_test = generate_output_pattern(
        BNN=approx_bnn, 
        X=X_test,
        gaussian_noise=False)
    save_data(X_test, Y_test, './data/', f'recurrent_test_nonoise.pkl')

    X_valid, Y_valid = generate_output_pattern(
        BNN=approx_bnn, 
        X=X_valid,
        gaussian_noise=False)
    save_data(X_valid, Y_valid, './data/', f'recurrent_valid_nonoise.pkl')