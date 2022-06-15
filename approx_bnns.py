import torch
import torch.nn as nn
import torch.distributions as dist

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForwardApproximateBNN(nn.Module):
    '''
    A feed-forward approximately biological network

    :param x:               hidden units in each layer
    :param y:               probability of randomly disabling weight (connection between each two neurons)
    :param z:               number of hidden layers
    '''
    def __init__(self, x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=True, trainable=False) -> None:
        super().__init__()
        
        self.layers = nn.Sequential()
        self.network_connectivity = x * y * z

        # input layer
        self.layers.add_module('input', nn.Linear(input_dim, x, bias=bias))
        self.layers.add_module('relu_1', transfer_function)
        
        # hidden layers
        for hidden_idx in range(1, z+1):
            self.layers.add_module(f'hidden_{hidden_idx}', nn.Linear(x, x, bias=bias))
            self.layers.add_module(f'relu_{hidden_idx+1}', transfer_function)

        # output layer
        self.layers.add_module('output', nn.Linear(x, output_dim, bias=bias))
        self.layers.add_module(f'ReLU_{z+1}', transfer_function)

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
    def __init__(self, x, y, z, input_dim, output_dim, residual_in, transfer_function=nn.ReLU(), bias=True, trainable=False):
        super().__init__()

        assert len(residual_in) == z

        self.z = z
        self.residual_in = residual_in
        self.transfer_function = transfer_function
        self.network_connectivity = x * y * (2 * z - sum([1 if i is not False else 0 for i in residual_in]))

        # input layer
        self.input_layer = nn.Linear(input_dim, x, bias=bias)

        # store hidden layers in a list
        self.hidden_layers = nn.ModuleList([])
        for hidden_idx in range(self.z):
            if self.residual_in[hidden_idx] == False:
                # if no residual connection, input_dim = x
                self.hidden_layers.append(nn.Linear(x, x, bias=bias))
            else:
                # if residual connection, input_dim = 2*x
                self.hidden_layers.append(nn.Linear(2*x, x, bias=bias))
        
        # output layer
        self.output_layer = nn.Linear(x, output_dim, bias=bias)
        
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
        x = self.transfer_function(x)

        temp_outputs = [None] * (self.z + 1)
        temp_outputs[0] = x

        for hidden_idx in range(self.z):

            if self.residual_in[hidden_idx] is False:
                # if there is no skip input, take input from prev layer
                temp_input = temp_outputs[hidden_idx]
                temp_outputs[hidden_idx+1] = self.hidden_layers[hidden_idx](temp_input)
                temp_outputs[hidden_idx+1] = self.transfer_function(temp_outputs[hidden_idx+1])

            else:
                # if there is skip input, concat the input with the prev layer
                temp_input = torch.concat((temp_outputs[hidden_idx], temp_outputs[self.residual_in[hidden_idx]]), dim=-1)
                temp_outputs[hidden_idx+1] = self.hidden_layers[hidden_idx](temp_input)
                temp_outputs[hidden_idx+1] = self.transfer_function(temp_outputs[hidden_idx+1])
        
        out = self.output_layer(temp_outputs[-1])
        out = self.transfer_function(out)
        
        return out


class RecurrentApproximateBNN(nn.Module):
    '''
    An approximately biological network with recurrent connections: hidden L4 -> hidden L1

    :param x:               hidden units in each layer
    :param y:               probability of randomly disabling weight (connection between each two neurons)
    :param z:               number of hidden layers
    :param recurrent_dim:   dimension of recurrent state. By default same as x
    '''
    def __init__(self, x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=True, trainable=False, recurrent_dim=-1):
        super().__init__()
        
        if recurrent_dim == -1: # if recurrent state dimension unspecified, default to same as hidden dim
            self.recurrent_dim = x
        else:
            self.recurrent_dim = recurrent_dim

        self.transfer_function = transfer_function
        self.output_dim = output_dim

        # input layer
        self.input_layer = nn.Linear(input_dim, x, bias=bias)

        # hidden layers, where the first hidden layer also takes in recurrent state
        self.hidden_layers = nn.Sequential()
        self.hidden_layers.add_module('hidden_1', nn.Linear(x + self.recurrent_dim, x, bias=bias))
        self.hidden_layers.add_module('relu_1', self.transfer_function)
        for hidden_idx in range(2, z+1):
            self.hidden_layers.add_module(f'hidden_{hidden_idx}', nn.Linear(x, x, bias=bias))
            self.hidden_layers.add_module(f'relu_{hidden_idx+1}', self.transfer_function)
        
        # output layer
        self.output_layer = nn.Linear(x, output_dim, bias=bias)

        # projection to hidden state
        self.recurrent_connection = nn.Linear(x, self.recurrent_dim, bias=bias)

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
            temp = self.transfer_function(temp) # (batch_size, hidden_dim)

            temp = torch.concat((temp, self.recurrent_state), dim=-1) # (batch_size, hidden_dim + recurrent_dim)
            temp = self.hidden_layers(temp)      # (batch_size, hidden_dim)

            yt = self.output_layer(temp)         # (batch_size, output_dim)
            y[:,t,:] = self.transfer_function(yt)
            
            ht = self.recurrent_connection(temp) # (batch_size, recurrent_dim)
            self.recurrent_state = self.transfer_function(ht)

        return y


if __name__ == '__main__':
    x = 64              # number of hidden units in each layer
    y = 0.5             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable

    residual_in =[False, False, 1, 2]

    input_dim = 16
    output_dim = 16

    approx_bnn = FeedForwardApproximateBNN(x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=bias, trainable=trainable)
    for name, params in approx_bnn.named_parameters():
        print(name, params.data)

    X = torch.distributions.Bernoulli(0.5).sample(sample_shape=[10000,16])
    approx_bnn(X)
