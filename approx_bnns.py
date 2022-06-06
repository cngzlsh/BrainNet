import torch
import torch.nn as nn
import torch.distributions as dist

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForwardApproximateBNN(nn.Module):
    '''
    A feed-forward approximately biological network
    '''
    def __init__(self, x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=True, trainable=False) -> None:
        super().__init__()
        
        self.layers = nn.Sequential()
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
        def apply_connectivity():
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    mask = dist.Bernoulli(probs=y).sample(sample_shape=layer.weight.shape)
                    layer.weight = nn.Parameter(torch.mul(layer.weight, mask))

                    if not trainable:
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
        
        apply_connectivity()

    def forward(self, input_pattern):
        return self.layers(input_pattern)


class RecurrentApproximateBNN(nn.Module):
    '''
    An approximately biological network with recurrent connections
    '''
    def __init__(self, x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=True, trainable=False):
        super().__init__()

        self.layers = nn.Sequential()  
        pass

class ResidualApproximateBNN(nn.Module):
    def __init__(self, x, y, z, input_dim, output_dim, residual, transfer_function=nn.ReLU(), bias=True, trainable=False):
        super().__init__()

        self.residual_out, self.residual_in = residual

        self.input_layer = nn.Linear(input_dim, x, bias=bias)
        self.transfer_function = transfer_function

        self.hidden_layers = []
        for _ in range(1, z+1):
            self.hidden_layers.append(nn.Linear(x, x, bias=True))
        
        self.output_layer = nn.Linear(x, output_dim, bias=True)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transfer_function(x)

        for i, layer in enumerate(self.hidden_layers):
            if i == self.residual_in:
                x += temp

            x = layer(x)
            x = self.transfer_function(x)

            if i == self.residual_out:
                temp = x

        return x


if __name__ == '__main__':
    x = 16              # number of hidden units in each layer
    y = 0.7             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    print(f'Network connectivity: {x * y * z}')

    input_dim = 1
    output_dim = 1

    approx_bnn = FeedForwardApproximateBNN(x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=bias, trainable=trainable).to(device)

    for name, params in approx_bnn.named_parameters():
        print(name, params.data)
