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
    
    def reset(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, input_pattern):
        return self.layers(input_pattern)
    

class RecurrentDNN(nn.Module):
    '''
    Deep neural network with LSTM units
    '''
    def __init__(self, input_dim, hidden_dim, n_linear_layers, output_dim, n_lstm_layers=1, transfer_function=nn.ReLU()):
        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.n_linear_layers = n_linear_layers
        self.hidden_dim = hidden_dim
        
        # input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.transfer_function = transfer_function

        # hidden layers
        self.hidden_lstms = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_lstm_layers, bias=True, batch_first=True, bidirectional=False)
        if self.n_linear_layers > 0:
            self.hidden_linears = nn.Sequential()
            for i in range(n_linear_layers):
                self.hidden_linears.add_module(f'hidden_linear_{i+1}', nn.Linear(hidden_dim, hidden_dim))
                self.hidden_linears.add_module(f'relu_{i+1}', self.transfer_function)

        # output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def reset(self):
        self.input_layer.reset_parameters()
        for layer in self.hidden_linears:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.output_layer.reset_parameters()
        self.hidden_lstms.reset_parameters()
    
    def forward(self, x, rec_prev):

        h_prev, c_prev = rec_prev

        x = self.input_layer(x)
        x = self.transfer_function(x)
        
        out, (h_curr, c_curr) = self.hidden_lstms(x, (h_prev, c_prev))

        if self.n_linear_layers > 0:
            out = self.hidden_linears(x)
    
        out = self.output_layer(x)
        return out, (h_curr, c_curr)
