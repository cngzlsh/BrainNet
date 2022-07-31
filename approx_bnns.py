from pandas import Categorical
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import random

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomActivation(nn.Module):
    '''
    Custom non-linearity layer, applies a random non-linearity for each dimension
    Mimics neuronal network where each neuron has slightly different activation
    transfer functions are ReLU(), SiLU(), LeakyReLU(), Sigmoid(), Tanh(), etc.
    '''
    def __init__(self, _dim, transfer_functions: list, max_fr=False):
        super().__init__()

        self.transfer_functions = random.choices(transfer_functions, k=_dim)
        self.dim = _dim

        if max_fr:
            self.max_output = dist.Poisson(rate=50).sample(sample_shape=torch.Size([_dim])).to(device)
            self.min_output = -dist.Poisson(rate=50).sample(sample_shape=torch.Size([_dim])).to(device)
        else:
            self.max_output = torch.Tensor([float('inf')]).to(device)
            self.min_output = -torch.Tensor([float('inf')]).to(device)

    def load_params(self, params:tuple):
        tr, max_fr, min_fr = params
        self.transfer_functions = tr
        self.max_output = max_fr.to(device)
        self.min_output = min_fr.to(device)

    def forward(self, x):
        # x: [batch_size, dim]
        unclipped_output = torch.stack([self.transfer_functions[i](x[:, i]) for i in range(self.dim)], dim=-1)
        return torch.maximum(
            torch.minimum(unclipped_output, self.max_output.repeat([unclipped_output.shape[0],1])),
                 self.min_output.repeat([unclipped_output.shape[0],1]))


class FeedForwardApproximateBNN(nn.Module):
    '''
    A feed-forward approximately biological network

    :param x:               hidden units in each layer
    :param y:               probability of randomly disabling weight (connection between each two neurons)
    :param z:               number of hidden layers
    '''
    def __init__(self, x, y, z, input_dim, output_dim, transfer_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(0.1), nn.SELU()], bias=True, trainable=False) -> None:
        super().__init__()
        
        self.layers = nn.Sequential()
        self.network_connectivity = x * y * z

        # input layer
        self.layers.add_module('input', nn.Linear(input_dim, x, bias=bias))
        self.layers.add_module('non_lin_1', RandomActivation(transfer_functions=transfer_functions, _dim=x))
        
        # hidden layers
        for hidden_idx in range(1, z+1):
            self.layers.add_module(f'hidden_{hidden_idx}', nn.Linear(x, x, bias=bias))
            self.layers.add_module(f'non_lin_{hidden_idx+1}', RandomActivation(transfer_functions=transfer_functions, _dim=x))

        # output layer
        self.layers.add_module('output', nn.Linear(x, output_dim, bias=bias))
        self.layers.add_module(f'non_lin_{z+2}', RandomActivation(transfer_functions=transfer_functions, _dim=output_dim))

        # neurons are connected ~Bernoulli(y)
        def apply_connectivity(): # makes the network more biologically plausible
            
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=1)
                    nn.init.normal_(layer.bias, mean=0, std=1)
                    
                    mask = dist.Bernoulli(probs=y).sample(sample_shape=layer.weight.shape)
                    layer.weight = nn.Parameter(torch.mul(layer.weight, mask))

                    if not trainable:
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
        
        apply_connectivity()

    def forward(self, input_pattern):
        n_data, ts, input_dim = input_pattern.shape
        input_pattern = input_pattern.view(-1, input_dim)
        return self.layers(input_pattern).view(n_data, ts, -1)


class ResidualApproximateBNN(nn.Module):
    '''
    A feed-forward approximately biological network with additional skip (residual) connections
    
    :param x:               hidden units in each layer
    :param y:               probability of randomly disabling weight (connection between each two neurons)
    :param z:               number of hidden layers
    :param residual_in:     list of len(z), denoting which layer (other than the prev layer) the input comes from. False if no skip connection as input
    '''
    def __init__(self, x, y, z, input_dim, output_dim, residual_in, transfer_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(0.1), nn.SELU()], bias=True, trainable=False):
        super().__init__()

        assert len(residual_in) == z

        self.z = z
        self.residual_in = residual_in

        # input layer
        self.input_layer = nn.Linear(input_dim, x, bias=bias)
        self.input_activation = RandomActivation(transfer_functions=transfer_functions, _dim=x)

        # store hidden layers in a list
        self.hidden_layers = nn.ModuleList([])
        self.hidden_activations = []
        for hidden_idx in range(self.z):
            if self.residual_in[hidden_idx] == False:
                # if no residual connection, input_dim = x
                self.hidden_layers.append(nn.Linear(x, x, bias=bias))
                self.hidden_activations.append(RandomActivation(transfer_functions=transfer_functions, _dim=x))
            else:
                # if residual connection, input_dim = 2*x
                self.hidden_layers.append(nn.Linear(2*x, x, bias=bias))
                self.hidden_activations.append(RandomActivation(transfer_functions=transfer_functions, _dim=x))
        
        # output layer
        self.output_layer = nn.Linear(x, output_dim, bias=bias)
        self.output_activation = RandomActivation(transfer_functions=transfer_functions, _dim=output_dim)
        
        def apply_connectivity(): # makes the network more biologically plausible
            # input layer
            nn.init.normal_(self.input_layer.weight, mean=0, std=1)
            nn.init.normal_(self.input_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.input_layer.weight.shape)
            self.input_layer.weight = nn.Parameter(torch.mul(self.input_layer.weight, mask))

            # hidden layers
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=1)
                    nn.init.normal_(layer.bias, mean=0, std=1)
                    mask = dist.Bernoulli(probs=y).sample(sample_shape=layer.weight.shape)
                    layer.weight = nn.Parameter(torch.mul(layer.weight, mask))

                    if not trainable:
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
            
            # output layer
            nn.init.normal_(self.output_layer.weight, mean=0, std=1)
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.output_layer.weight.shape)
            self.output_layer.weight = nn.Parameter(torch.mul(self.output_layer.weight, mask))

            if not trainable:
                self.input_layer.weight.requires_grad = False
                self.input_layer.bias.requires_grad = False
                self.output_layer.weight.requires_grad = False
                self.output_layer.bias.requires_grad = False

        apply_connectivity()

    def forward(self, input_pattern): 
        n_data, ts, input_dim = input_pattern.shape
        input_pattern = input_pattern.view(-1, input_dim)


        input_pattern = self.input_layer(input_pattern)
        input_pattern = self.input_activation(input_pattern)

        temp_outputs = [None] * (self.z + 1)
        temp_outputs[0] = input_pattern

        for hidden_idx in range(self.z):

            if self.residual_in[hidden_idx] is False:
                # if there is no skip input, take input from prev layer
                temp_input = temp_outputs[hidden_idx]
                temp_outputs[hidden_idx+1] = self.hidden_layers[hidden_idx](temp_input)
                temp_outputs[hidden_idx+1] = self.hidden_activations[hidden_idx](temp_outputs[hidden_idx+1])

            else:
                # if there is skip input, concat the input with the prev layer
                temp_input = torch.concat((temp_outputs[hidden_idx], temp_outputs[self.residual_in[hidden_idx]]), dim=-1)
                temp_outputs[hidden_idx+1] = self.hidden_layers[hidden_idx](temp_input)
                temp_outputs[hidden_idx+1] = self.hidden_activations[hidden_idx](temp_outputs[hidden_idx+1])
        
        out = self.output_layer(temp_outputs[-1])
        out = self.output_activation(out)
        
        return out.view(n_data, ts, -1)


class RecurrentApproximateBNN(nn.Module):
    '''
    An approximately biological network with recurrent connections: hidden L4 -> hidden L1
8.4794e+04, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.3830e+05, 1.0001e+05,
         1.0000e+00, 8.6347e+04, 1.2372e+05, 8.6820e+04, 1.0000e+00, 5.2773e+04,
         1.1521e+05, 1.2773e+05, 6.1236e+04, 1.0000e+00            hidden units in each layer
    :param y:               probability of randomly disabling weight (connection between each two neurons)
    :param z:               number of hidden layers
    :param recurrent_dim:   dimension of recurrent state. By default same as x
    '''
    def __init__(self, x, y, z, input_dim, output_dim, recurrent_dim=-1, transfer_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(0.1), nn.SELU()], bias=True, trainable=False):
        super().__init__()
        
        if recurrent_dim == -1: # if recurrent state dimension unspecified, default to same as hidden dim
            self.recurrent_dim = x
        else:
            self.recurrent_dim = recurrent_dim

        self.output_dim = output_dim

        # input layer
        self.input_layer = nn.Linear(input_dim, x, bias=bias)
        self.input_activation = RandomActivation(transfer_functions=transfer_functions, _dim=x)

        # hidden layers, where the first hidden layer also takes in recurrent state
        self.hidden_layers = nn.Sequential()
        self.hidden_layers.add_module('hidden_1', nn.Linear(x + self.recurrent_dim, x, bias=bias))
        self.hidden_layers.add_module('non_lin_1', RandomActivation(transfer_functions=transfer_functions, _dim=x))
        for hidden_idx in range(2, z+1):
            self.hidden_layers.add_module(f'hidden_{hidden_idx}', nn.Linear(x, x, bias=bias))
            self.hidden_layers.add_module(f'non_lin_{hidden_idx+1}', RandomActivation(transfer_functions=transfer_functions, _dim=x))
        
        # output layer
        self.output_layer = nn.Linear(x, output_dim, bias=bias)
        self.output_activation = RandomActivation(transfer_functions=transfer_functions, _dim=output_dim)

        def apply_connectivity(): # makes the network more biologically plausible
            # input layer
            nn.init.normal_(self.input_layer.weight, mean=0, std=1)
            nn.init.normal_(self.input_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.input_layer.weight.shape)
            self.input_layer.weight = nn.Parameter(torch.mul(self.input_layer.weight, mask))

            # hidden layers
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=1)
                    nn.init.normal_(layer.bias, mean=0, std=1)
                    mask = dist.Bernoulli(probs=y).sample(sample_shape=layer.weight.shape)
                    layer.weight = nn.Parameter(torch.mul(layer.weight, mask))

                    if not trainable:
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
            
            # output layer
            nn.init.normal_(self.output_layer.weight, mean=0, std=1)
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.output_layer.weight.shape)
            self.output_layer.weight = nn.Parameter(torch.mul(self.output_layer.weight, mask))

            if not trainable:
                self.input_layer.weight.requires_grad = False
                self.input_layer.bias.requires_grad = False
                self.output_layer.weight.requires_grad = False
                self.output_layer.bias.requires_grad = False
        
        apply_connectivity()
    
    def forward(self, x):
        '''
        Pass a temporal sequence through RNN. input is of shape (batch_size, num_time_steps, input_dim)
        '''
        batch_size, time_steps, _ = x.shape

        # initialise recurrent state and output tensor
        self.recurrent_state = torch.zeros(batch_size, self.recurrent_dim).to(device)
        y = torch.zeros([batch_size, time_steps, self.output_dim]).to(device)

        for t in range(time_steps):
            
            temp = self.input_layer(x[:,t,:]) # (batch_size, hidden_dim)
            temp = self.input_activation(temp) # (batch_size, hidden_dim)

            temp = torch.concat((temp, self.recurrent_state), dim=-1) # (batch_size, hidden_dim + recurrent_dim)
            temp = self.hidden_layers(temp)      # (batch_size, hidden_dim)

            self.recurrent_state = F.normalize(temp)

            yt = self.output_layer(temp)         # (batch_size, output_dim)
            y[:,t,:] = self.output_activation(yt)

        return y



class ComplexApproximateBNN(nn.Module):
    '''
    An (arbitrarily complex) approximately biological network with
        residual (skip) connection:     hidden L1 -> hidden L3
                                        hidden L2 -> hidden L4
        lateral inhibition:             hidden L2 -> hidden L2
        backprojection:                 hidden L4 -> hidden L1

    :param x:               hidden units in each layer
    :param y:               probability of randomly disabling weight (connection between each two neurons)
    :param z:               number of hidden layers
    :param residual_in:     list of len(z), denoting which layer (other than the prev layer) the input comes from. False if no skip connection as input
    :param recurrent_dim:   dimension of recurrent state. By default same as x
    '''
    def __init__(self, x, y, z, input_dim, output_dim, residual_in, recurrent_dim=-1, transfer_functions=[nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(0.1), nn.SELU()], bias=True, trainable=False):
        super().__init__()

        assert len(residual_in) == z
        self.x = x
        self.z = z
        self.output_dim = output_dim
        self.residual_in = residual_in
        self.recurrent_dim = x if recurrent_dim == -1 else recurrent_dim
        self.activations = [RandomActivation(_dim=x, transfer_functions=transfer_functions) for _ in range(z+1)] # input layer and z hidden layers

        # input layer
        self.input_layer = nn.Linear(input_dim, x, bias=bias)

        # store hidden layers in a list
        self.hidden_layers = nn.ModuleList([])
        for _ in range(self.z):
                # hidden 1: recurrent from hidden 4
                # hidden 2: recurrent from layer 2
                # hidden 3: skip from layer 1
                # hidden 4: skip from layer 2
                self.hidden_layers.append(nn.Linear(2*x, x, bias=bias))

        # output layer
        self.output_layer = nn.Linear(x, output_dim, bias=bias)
        self.output_activation = RandomActivation(_dim=output_dim, transfer_functions=transfer_functions)

        def apply_connectivity():

            # input layer
            nn.init.normal_(self.input_layer.weight, mean=0, std=1)
            nn.init.normal_(self.input_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.input_layer.weight.shape)
            self.input_layer.weight = nn.Parameter(torch.mul(self.input_layer.weight, mask))

            # lateral inhibition weights
            temp = torch.concat((torch.linspace(0, 0, x+1)[:-1], torch.linspace(0, 0, x+1)), dim=0)
            lateral_inhibition_weights = dist.Normal(0, 1).sample(sample_shape=torch.Size([x,x]))
            for i in range(x):
                lateral_inhibition_weights[i,:] += temp[x-i: 2*x-i]
            lateral_inhibition_weights.fill_diagonal_(0) # no inhibition on itself

            # hidden layers
            for i, layer in enumerate(self.hidden_layers):
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=1)
                    nn.init.normal_(layer.bias, mean=0, std=1)
                    if i == 1:
                        layer.weight.data[:, x:] = lateral_inhibition_weights
                        
                    mask = dist.Bernoulli(probs=y).sample(sample_shape=layer.weight.shape)
                    layer.weight = nn.Parameter(torch.mul(layer.weight, mask))

                    if not trainable:
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False

            # output layer
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.output_layer.weight.shape)
            self.output_layer.weight = nn.Parameter(torch.mul(self.output_layer.weight, mask))

            if not trainable:
                self.input_layer.weight.requires_grad = False
                self.input_layer.bias.requires_grad = False
                self.output_layer.weight.requires_grad = False
                self.output_layer.bias.requires_grad = False
        
        apply_connectivity()
    
    def extract_non_linearities(self):
        '''
        Saves RandomActivation() parameters to a dictionary
        '''
        _dict = {}

        for i, activation_function in enumerate(self.activations):
            _dict[str(i)] = (activation_function.transfer_functions, activation_function.max_output.cpu(), activation_function.min_output.cpu())
        _dict['output'] = (self.output_activation.transfer_functions, self.output_activation.max_output.cpu(), self.output_activation.min_output.cpu())

        return _dict

    def load_non_linearities(self, _dict):
        '''
        Loads RandomActivation parameters from a dictionary
        '''
        for i, activation_function in enumerate(self.activations):
            activation_function.load_params(_dict[str(i)])
        self.output_activation.load_params(_dict['output'])


    def gaussian_plasticity_update(self, sigma, alpha):
        '''
        Mimics neuronal plasticity dynamics, slightly alter each non-zero weight by injecting a small Gaussian noise to all layers

        params:
        sigma:      scalar, std of Gaussian noise
        alpha:      scalar between 0 and 1. Proportion of weights to change
        '''
        # input layer
        non_zero_mask = torch.multiply(
            dist.Bernoulli(probs=alpha).sample(sample_shape=self.input_layer.weight.data.shape).to(device), 
            (self.input_layer.weight.data != 0))
        # gaussian_noise = torch.zeros_like(self.input_layer.weight.data)
        gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=self.input_layer.weight.data.shape).to(device)
        
        self.input_layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))

        # hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                non_zero_mask = torch.multiply(
                    dist.Bernoulli(probs=alpha).sample(sample_shape=layer.weight.data.shape).to(device),
                    (layer.weight.data != 0))
                gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=layer.weight.data.shape).to(device)
                # gaussian_noise = torch.zeros_like(layer.weight.data)
                layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))
        
        # output layer
        non_zero_mask = torch.multiply(
            dist.Bernoulli(probs=alpha).sample(sample_shape=self.output_layer.weight.data.shape).to(device), 
            (self.output_layer.weight.data != 0))

        gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=self.output_layer.weight.data.shape).to(device)
        # gaussian_noise = torch.zeros_like(self.output_layer.weight.data)
        self.output_layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))

    
    def forward(self, x):
        '''
        Pass a temporal sequence through complex BNN. input is of shape (batch_size, num_time_steps, input_dim)
        '''
        batch_size, time_steps, _ = x.shape

        # initialise recurrent states and output tensor
        self.backprojection_state = torch.zeros(batch_size, self.recurrent_dim).to(device)
        self.lateral_inhibition_state = torch.zeros(batch_size, self.x).to(device)
        
        y = torch.zeros([batch_size, time_steps, self.output_dim]).to(device)

        for t in range(time_steps):
            
            temp = self.input_layer(x[:,t,:]) # (batch_size, hidden_dim)
            temp = self.activations[0](temp) # (batch_size, hidden_dim)

            temp_outputs = [None] * (self.z + 1)
            temp_outputs[0] = temp

            for hidden_idx in range(self.z):
                
                if hidden_idx == 0: # first hidden layer receives recurrent connection
                    temp = torch.concat((temp_outputs[0], self.backprojection_state), dim=-1) # (batch_size, hidden_dim + recurrent_dim)
                    temp_outputs[hidden_idx+1] = self.activations[hidden_idx+1](self.hidden_layers[0](temp))     # (batch_size, hidden_dim)
                
                if hidden_idx == 1: # second hidden layer receives lateral inhibition
                    temp = torch.concat((temp_outputs[1], self.lateral_inhibition_state), dim=-1)
                    temp_outputs[hidden_idx+1] = self.activations[hidden_idx+1](self.hidden_layers[1](temp))
                    self.lateral_inhibition_state = F.normalize(temp_outputs[hidden_idx+1])

                if hidden_idx not in set([0,1]):
                    if self.residual_in[hidden_idx]:
                    # if there is skip input, concat the input with the prev layer
                        temp_input = torch.concat((temp_outputs[hidden_idx], temp_outputs[self.residual_in[hidden_idx]]), dim=-1)
                        temp_outputs[hidden_idx+1] = self.hidden_layers[hidden_idx](temp_input)
                        temp_outputs[hidden_idx+1] = self.activations[hidden_idx+1](temp_outputs[hidden_idx+1])
                    
                    else:
                     # if there is no skip input, take input from prev layer
                        temp_input = temp_outputs[hidden_idx]
                        temp_outputs[hidden_idx+1] = self.hidden_layers[hidden_idx](temp_input)
                        temp_outputs[hidden_idx+1] = self.activations[hidden_idx+1](temp_outputs[hidden_idx+1])

            self.backprojection_state = F.normalize(temp_outputs[-1])
            
            y[:,t,:] = self.output_activation(self.output_layer(temp_outputs[-1]))

        return y
