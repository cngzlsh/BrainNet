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
    def __init__(self, _dim, transfer_functions: list):
        super().__init__()

        self.transfer_functions = random.choices(transfer_functions, k=_dim)
        self.dim = _dim
        self.max_output = dist.Poisson(rate=10).sample(sample_shape=torch.Size([_dim])).to(device) # not sure what to set the max as??
    
    def load_params(self, key):
        if isinstance(key, torch.Tensor):
            assert key.shape == self.max_output.shape
            self.max_output = key   # loads max_output
        else:
            assert len(key) == len(self.transfer_functions)
            self.transfer_functions = key   # loaders transfer functions

    def forward(self, x):
        # x: [batch_size, dim]
        unclipped_output = torch.stack([self.transfer_functions[i](x[:, i]) for i in range(self.dim)], dim=-1)
        return torch.minimum(unclipped_output, self.max_output.repeat([unclipped_output.shape[0],1]))


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
        return self.layers(input_pattern)


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

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)

        temp_outputs = [None] * (self.z + 1)
        temp_outputs[0] = x

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
        
        return out


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

        # projection to hidden state
        self.recurrent_connection = nn.Linear(x, self.recurrent_dim, bias=bias)
        self.recurrent_activation = RandomActivation(transfer_functions=transfer_functions, _dim=x)

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

            # recurrent connection
            nn.init.normal_(self.recurrent_connection.weight, mean=0, std=1)
            nn.init.normal_(self.recurrent_connection.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.recurrent_connection.weight.shape)
            self.recurrent_connection.weight = nn.Parameter(torch.mul(self.recurrent_connection.weight, mask))

            if not trainable:
                self.input_layer.weight.requires_grad = False
                self.input_layer.bias.requires_grad = False
                self.output_layer.weight.requires_grad = False
                self.output_layer.bias.requires_grad = False
                self.recurrent_connection.weight.requires_grad = False
                self.recurrent_connection.bias.requires_grad = False
        
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

            self.recurrent_state = F.normalize(self.recurrent_state) # normalise recurrent state to ensure no nan

            temp = torch.concat((temp, self.recurrent_state), dim=-1) # (batch_size, hidden_dim + recurrent_dim)
            temp = self.hidden_layers(temp)      # (batch_size, hidden_dim)

            yt = self.output_layer(temp)         # (batch_size, output_dim)
            y[:,t,:] = self.output_activation(yt)
            
            ht = self.recurrent_connection(temp) # (batch_size, recurrent_dim)
            self.recurrent_state = self.recurrent_activation(ht)

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
        self.lateral_inhibition_activation = RandomActivation(_dim=x, transfer_functions=transfer_functions)
        self.recurrent_activation = RandomActivation(_dim = self.recurrent_dim, transfer_functions=transfer_functions)
        self.output_activation = RandomActivation(_dim=output_dim, transfer_functions=transfer_functions)

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
        
        # lateral inhibition layer
        self.lateral_inhibition_layer = nn.Linear(x, x, bias=bias)

        # output layer
        self.output_layer = nn.Linear(x, output_dim, bias=bias)

        # backprojection
        self.recurrent_connection = nn.Linear(x, self.recurrent_dim, bias=bias)

        def apply_connectivity():

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
            
            # lateral inhibition layer
            nn.init.normal_(self.lateral_inhibition_layer.weight, mean=-1, std=1)
            nn.init.normal_(self.lateral_inhibition_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.lateral_inhibition_layer.weight.shape)
            self.lateral_inhibition_layer.weight = nn.Parameter(torch.mul(self.lateral_inhibition_layer.weight, mask))

            # output layer
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
            nn.init.normal_(self.output_layer.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.output_layer.weight.shape)
            self.output_layer.weight = nn.Parameter(torch.mul(self.output_layer.weight, mask))

            # backprojection
            nn.init.normal_(self.output_layer.bias, mean=-1, std=1)
            nn.init.normal_(self.recurrent_connection.bias, mean=0, std=1)
            mask = dist.Bernoulli(probs=y).sample(sample_shape=self.recurrent_connection.weight.shape)
            self.recurrent_connection.weight = nn.Parameter(torch.mul(self.recurrent_connection.weight, mask))

            if not trainable:
                self.input_layer.weight.requires_grad = False
                self.input_layer.bias.requires_grad = False
                self.output_layer.weight.requires_grad = False
                self.output_layer.bias.requires_grad = False
                self.lateral_inhibition_layer.weight.requires_grad = False
                self.lateral_inhibition_layer.bias.requires_grad = False
                self.recurrent_connection.weight.requires_grad = False
                self.recurrent_connection.bias.requires_grad = False
        
        apply_connectivity()
    
    def save_non_linearities(self):
        '''
        Saves RandomActivation to a dictionary
        '''
        _dict = {}
        for name, module in self.named_modules():
            if isinstance(module, RandomActivation):
                _dict[name] = module.transfer_functions
                _dict[name+'_max_fr'] = module.max_output
        return _dict

    def load_non_linearities(self, _dict):
        '''
        Loads RandomActivation from a dictionary
        '''
        for name, module in self.named_modules():
            if isinstance(module, RandomActivation):
                module.load_params(_dict[name])
                module.load_params(_dict[name+'_max_fr'])
        print('Non-linearities loaded successfully.')


    def gaussian_plasticity_update(self, sigma, alpha=1):
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
        gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=self.input_layer.weight.data.shape).to(device)
        self.input_layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))

        # hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                non_zero_mask = torch.multiply(
                    dist.Bernoulli(probs=alpha).sample(sample_shape=layer.weight.data.shape).to(device),
                    (layer.weight.data != 0))
                gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=layer.weight.data.shape).to(device)
                layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))
        
        # output layer
        non_zero_mask = torch.multiply(
            dist.Bernoulli(probs=alpha).sample(sample_shape=self.output_layer.weight.data.shape).to(device), 
            (self.output_layer.weight.data != 0))
        gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=self.output_layer.weight.data.shape).to(device)
        self.output_layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))

        # lateral inhibition layer
        non_zero_mask = torch.multiply(
            dist.Bernoulli(probs=alpha).sample(sample_shape=self.lateral_inhibition_layer.weight.data.shape).to(device), 
            (self.lateral_inhibition_layer.weight.data != 0))
        gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=self.lateral_inhibition_layer.weight.data.shape).to(device)
        self.lateral_inhibition_layer.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))

        # backprojection
        non_zero_mask = non_zero_mask = torch.multiply(
            dist.Bernoulli(probs=alpha).sample(sample_shape=self.recurrent_connection.weight.data.shape).to(device), 
            (self.recurrent_connection.weight.data != 0))
        gaussian_noise = dist.Normal(loc=0, scale=sigma).sample(sample_shape=self.recurrent_connection.weight.data.shape).to(device)
        self.recurrent_connection.weight.data += nn.Parameter(torch.mul(gaussian_noise, non_zero_mask))
    
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

            # self.backprojection_state = F.normalize(self.backprojection_state) # normalise recurrent state to ensure no nan
            # self.lateral_inhibition_state = F.normalize(self.lateral_inhibition_state)

            for hidden_idx in range(self.z):
                
                if hidden_idx == 0: # first hidden layer receives recurrent connection
                    temp = torch.concat((temp_outputs[0], self.backprojection_state), dim=-1) # (batch_size, hidden_dim + recurrent_dim)
                    temp_outputs[hidden_idx+1] = self.activations[hidden_idx+1](self.hidden_layers[0](temp))     # (batch_size, hidden_dim)
                
                if hidden_idx == 1: # second hidden layer receives lateral inhibition
                    temp = torch.concat((temp_outputs[1], self.lateral_inhibition_state), dim=-1)
                    temp_outputs[hidden_idx+1] = self.activations[hidden_idx+1](self.hidden_layers[1](temp))
                    self.lateral_inhibition_state = self.lateral_inhibition_activation(self.lateral_inhibition_layer(temp_outputs[hidden_idx+1]))

                if hidden_idx not in [0,1]:
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
            self.backprojection_state = self.recurrent_activation(self.recurrent_connection(temp_outputs[-1])) # (batch_size, recurrent_dim)
            y[:,t,:] = self.output_activation(self.output_layer(temp_outputs[-1]))

        return y
