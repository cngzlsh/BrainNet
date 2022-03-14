import torch
import torch.nn as nn
import torch.distributions as dist

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeedForwardApproximateBNN(nn.Module):
    '''
    An approximately biological network
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
    pass

class ResidualApproximateBNN(nn.Module):
    pass

if __name__ == '__main__':
    x = 16              # number of hidden units in each layer
    y = 0.9             # network connectivity
    z = 4               # number of layers
    bias = True         # whether to use bias
    trainable = False   # whether the network is trainable
    print(f'Network connectivity: {x * y * z}')

    input_dim = 1
    output_dim = 1

    approx_bnn = ApproximateBNN(x, y, z, input_dim, output_dim, transfer_function=nn.ReLU(), bias=bias, trainable=trainable).to(device)

    for param in approx_bnn.parameters():
        print(param.data)
