import torch
import torch.nn as nn


class FeedForwardDNN(nn.Module):
    '''
    Feed-forward deep neural network, resembling the structure of FeedForwardApproximateBNN, but no dropouts
    '''
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, transfer_function=nn.ReLU()):
        super().__init__()
        
        self.layers = nn.Sequential()

        # input layer
        self.layers.add_module('input', nn.Linear(input_dim, hidden_dim))
        self.layers.add_module('relu_1', transfer_function)
        
        # hidden layers
        for hidden_idx in range(1, n_layers+1):
            self.layers.add_module(f'hidden_{hidden_idx}', nn.Linear(hidden_dim, hidden_dim))
            self.layers.add_module(f'relu_{hidden_idx+1}', transfer_function)

        # output layer
        self.layers.add_module('output', nn.Linear(hidden_dim, output_dim))
        self.layers.add_module(f'ReLU_{n_layers+1}', transfer_function)

    def forward(self, input_pattern):
        return self.layers(input_pattern)