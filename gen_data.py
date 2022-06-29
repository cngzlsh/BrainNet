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
    Y = normalise_data(Y)

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
    Y = normalise_data(Y)

    if gaussian_noise is not False:
        mu, sigma = gaussian_noise
        Y += dist.Normal(loc=mu, scale=sigma).sample(sample_shape=Y.shape).to(device)[:,:,0]
    
    return X.cpu(), Y.cpu()


def apply_plasticity_and_generate_new_output(BNN, BNN_weights, X_train, X_test, sigma, verbose=True):
    '''
    Slightly alter each non-zero weight in the biological neural network, and generate new output patterns
    '''
    
    BNN.load_state_dict(torch.load(BNN_weights))
    BNN.gaussian_plasticity_update(sigma)
    
    if verbose:
        print('\t Plasticity applied.')
    
    Y_train = normalise_data(BNN(X_train))
    Y_test = normalise_data(BNN(X_test))

    if verbose:
        print('\t New output patterns generated')
    return Y_train, Y_test


def generate_bvc_network_firing_pattern(n_data_points, n_cells, preferred_distances, preferred_orientations, sigma_rads, sigma_angs):
    
    BVCs = [BVC(r= preferred_distances[i], theta=preferred_orientations[i], sigma_rad=sigma_rads[i], sigma_ang=sigma_angs[i], scaling_factor=1) for i in range(n_cells)]
    model = BVCNetwork(BVCs=BVCs, coeff=1, threshold=0, non_linearity=nn.ReLU())

    ds = dist.uniform.Uniform(low=0, high=10).sample(sample_shape=torch.Size([n_data_points]))
    phis = dist.uniform.Uniform(low=-torch.pi, high=torch.pi).sample(sample_shape=torch.Size([n_data_points]))

    X = torch.stack((ds, phis), dim=-1)
    Y = model.obtain_firing_rate(ds, phis)[:, None]
    return X.cpu(), Y.cpu()


if __name__ == '__main__':
    x = 256            # number of hidden units in each layer
    y = 0.5             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    residual_in = [False, False, 1, 2]

    input_dim = 16
    output_dim = 16
    num_train = 10000
    num_test = 1000
    num_valid = 4000
    mean_freq = 5

    alphas = torch.ones(input_dim) * 2
    betas = dist.Poisson(rate=mean_freq*2).sample(sample_shape=torch.Size([input_dim]))

    time_steps = 50
    gaussian_noise = (torch.Tensor([0]), torch.Tensor([0.001]))
    transfer_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(0.1), nn.SELU()]

    approx_bnn = ComplexApproximateBNN(
        x=x, y=y, z=z,
        input_dim=input_dim,
        output_dim=output_dim, 
        residual_in=residual_in,
        recurrent_dim=-1,
        transfer_functions=transfer_functions
        ).to(device)

    torch.save(approx_bnn.state_dict(), f'./approx_bnn_params/cplx_abnn_{x}_{y}_{z}.pt')
    save_non_linearities(approx_bnn.save_non_linearities(), './approx_bnn_params/', f'cplx_abnn_{x}_{y}_{z}_random_activation.pkl')
    # approx_bnn.gaussian_weight_update(sigma=0.1)


    X_train, Y_train = generate_time_dependent_stochastic_pattern(BNN=approx_bnn, input_dim=input_dim, n_data_points=num_train, alphas=alphas, betas=betas, time_steps=time_steps, gaussian_noise=False)
    # save_data(X_train, Y_train, './data/', f'cplx_train_abnn_{x}_{y}_{z}_2.pkl')

    X_test, Y_test = generate_time_dependent_stochastic_pattern(BNN=approx_bnn, input_dim=input_dim, n_data_points=num_test, alphas=alphas, betas=betas, time_steps=time_steps, gaussian_noise=False)
    # save_data(X_test, Y_test, './data/', f'cplx_test_abnn_{x}_{y}_{z}_2.pkl')

    X_valid, Y_valid = generate_time_dependent_stochastic_pattern(BNN=approx_bnn, input_dim=input_dim, n_data_points=num_valid, alphas=alphas, betas=betas, time_steps=time_steps, gaussian_noise=False)
    # save_data(X_valid, Y_valid, './data/', f'cplx_valid_abnn_{x}_{y}_{z}_2.pkl')

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