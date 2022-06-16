import torch
import torch.nn as nn


class FeedForwardDNN(nn.Module):
    '''
    Feed-forward deep neural network
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

    def forward(self, input_pattern):
        return self.layers(input_pattern)
    
class RecurrentDNN(nn.Module):
    '''
    Deep neural network with LSTM units
    '''
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, transfer_function=nn.ReLU()):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # hidden LSTM layers
        self.hidden_lstms = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.hidden_lstms.append(nn.LSTM(hidden_dim, hidden_dim, batch_first=True))
        
        # output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, rec_prev):

        h_prev, c_prev = rec_prev

        x = self.input_layer(x)
        h_curr, c_curr = torch.zeros_like(h_prev), torch.zeros_like(c_prev)

        for i in range(self.n_layers):
            x, (h_i, c_i) = self.hidden_lstms[i](x, (h_prev, c_prev))
            h_curr[i] = h_i
            c_curr[i] = c_i
        
        out = self.output_layer(x)

        return out, (h_curr, c_curr)
