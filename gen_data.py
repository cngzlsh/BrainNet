from approx_bnns import *
from bvc import *
from utils import *

import torch
import torch.nn as nn
import torch.distributions as dist

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalise_data(data):
    '''
    Normalises data: subtract mean and divide by std over batch_size dim (dim 0)
    '''
    return torch.nan_to_num((data - torch.mean(data, dim=0))/ torch.std(data, dim=0))

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
    Y = normalise_data(Y)

    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]

    return X.cpu(), Y.cpu()


def generate_stochastic_firing_pattern(BNN, input_dim, n_data_points, mean_freq, gaussian_noise=False):
    '''
    A simple, approximately biological input pattern: {input_dim} input neurons into the approximate neuronal network,
    the input is the number of spikes in a very small time bin. The number of spikes for each presynaptic neuron follows
    independent Poisson distribution.

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


def generate_time_dependent_stochastic_pattern(BNN, input_dim, n_data_points, alphas, betas, time_steps=50, gaussian_noise=False):
    '''
    Generates time series of patterns
    '''
    if isinstance(alphas, int) or isinstance(alphas, float):
        alphas = torch.Tensor(torch.ones(input_dim) * alphas)
        betas = torch.Tensor(torch.ones(input_dim) * betas)

    X = torch.stack([gen_multiple_spike_train_counts(alpha=alphas, beta=betas, time_steps=time_steps) for _ in range( n_data_points)]).to(device)

    Y = BNN(X)

    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]

    Y = normalise_data(Y)
    
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
    
    _Y_train = normalise_data(BNN(X_train.to(device)))
    _Y_test = normalise_data(BNN(X_test.to(device)))
    
    if verbose:
        print('\t New output patterns generated.')

    return _Y_train, _Y_test


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
    num_valid = 4000
    mean_freq = 5

    alphas = torch.ones(input_dim) * 2
    betas = dist.Poisson(rate=mean_freq*2).sample(sample_shape=torch.Size([input_dim]))

    time_steps = 50
    transfer_functions=[nn.ReLU(), nn.ELU(), nn.SiLU(), nn.CELU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(0.1), nn.LeakyReLU(0.2), nn.LeakyReLU(0.3)]

    approx_bnn = ComplexApproximateBNN(
        x=x, y=y, z=z,
        input_dim=input_dim,
        output_dim=output_dim, 
        residual_in=residual_in,
        recurrent_dim=-1,
        transfer_functions=transfer_functions
        ).to(device)


    torch.save(approx_bnn.state_dict(), './approx_bnn_params/complex.pt')
    save_non_linearities(approx_bnn.extract_non_linearities(), './approx_bnn_params/complex_activations.pkl')


    X_train, Y_train = generate_time_dependent_stochastic_pattern(
        BNN=approx_bnn, 
        input_dim=input_dim, 
        n_data_points=num_train, 
        alphas=alphas, 
        betas=betas, 
        # mean_freq=mean_freq,
        time_steps=time_steps,
        gaussian_noise=False)
    save_data(X_train, Y_train, './data/', f'complex_train.pkl')

    X_test, Y_test = generate_time_dependent_stochastic_pattern(
        BNN=approx_bnn, 
        input_dim=input_dim,
        n_data_points=num_test, 
        alphas=alphas, 
        betas=betas, 
        # mean_freq=mean_freq,
        time_steps=time_steps, 
        gaussian_noise=gaussian_noise)
    save_data(X_test, Y_test, './data/', f'complex_test.pkl')

    X_valid, Y_valid = generate_time_dependent_stochastic_pattern(
        BNN=approx_bnn, 
        input_dim=input_dim, 
        n_data_points=num_valid, 
        alphas=alphas, 
        betas=betas, 
        # mean_freq=mean_freq,
        time_steps=time_steps, 
        gaussian_noise=gaussian_noise)
    save_data(X_valid, Y_valid, './data/', f'complex_valid.pkl')

    # Y_train_s, _ = apply_plasticity_and_generate_new_output(approx_bnn, (BNN_weights, BNN_non_linearities), X_train, X_test, sigma=0.0002, alpha=1, verbose=True)