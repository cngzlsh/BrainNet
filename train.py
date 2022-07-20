from re import A
import torch
import torch.nn as nn
import numpy as np
from models import *
from utils import *
from torch.utils.data import DataLoader, RandomSampler
from gen_data import apply_plasticity_and_generate_new_output

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False, return_init_eval_loss=False):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss.
    '''
    
    start = time.time()

    eval_losses, train_losses = [], []

    init_eval_loss = eval(model, test_loader, criterion)
    if verbose:
        print(f'Initial eval loss: {init_eval_loss}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (X, Y) in enumerate(iter(train_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

            if force_stop and i == 20:
                break

        eval_loss = eval(model, test_loader, criterion)

        train_losses.append(epoch_loss)
        eval_losses.append(eval_loss)

        if verbose:
            epoch_end = time.time()
            if num_epochs < 50:
                hrs, mins, secs = elapsed_time(start, epoch_end)
                print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
            else:
                if epoch % 20 == 0:
                    hrs, mins, secs = elapsed_time(start, epoch_end)
                    print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
    
    hrs, mins, secs = elapsed_time(start, time.time())

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
    
    if return_init_eval_loss:
        return train_losses, eval_losses, init_eval_loss
    else:
        return train_losses, eval_losses,


def eval(model, test_loader, criterion):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()

    with torch.no_grad():

        eval_loss = 0

        for _, (X, Y) in enumerate(iter(test_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    return eval_loss


def train_rnn(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False, return_init_eval_loss=False):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss.
    '''
    start = time.time()

    eval_losses, train_losses = [], []

    init_eval_loss = eval_rnn(model, test_loader, criterion)
    if verbose:
        print(f'Initial eval loss: {init_eval_loss}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        h_prev = torch.zeros([model.n_lstm_layers, train_loader.batch_size, model.hidden_dim]).to(device)
        c_prev = torch.zeros([model.n_lstm_layers, train_loader.batch_size, model.hidden_dim]).to(device)
        rec_prev = (h_prev, c_prev)

        for i, (X, Y) in enumerate(iter(train_loader)): # X: [batch_size, time_step, input_dim]
            
            Y_hat, rec_prev = model(X, rec_prev)
            loss = criterion(Y_hat, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

            if force_stop and i == 20:
                break

        eval_loss = eval_rnn(model, test_loader, criterion)

        train_losses.append(epoch_loss)
        eval_losses.append(eval_loss)

        if verbose:
            epoch_end = time.time()
            if num_epochs < 50:
                hrs, mins, secs = elapsed_time(start, epoch_end)
                print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
            else:
                if epoch % 20 == 0:
                    hrs, mins, secs = elapsed_time(start, epoch_end)
                    print(f'Epoch {epoch+1}: training loss {epoch_loss}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
    
    hrs, mins, secs = elapsed_time(start, time.time())

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
    
    if return_init_eval_loss:
        return train_losses, eval_losses, init_eval_loss
    else:
        return train_losses, eval_losses


def eval_rnn(model, test_loader, criterion, save_Y_hat=False):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()

    if save_Y_hat:
        Y_hats = []

    with torch.no_grad():

        eval_loss = 0
        
        h_prev = torch.zeros([model.n_lstm_layers, test_loader.batch_size, model.hidden_dim]).to(device)
        c_prev = torch.zeros([model.n_lstm_layers, test_loader.batch_size, model.hidden_dim]).to(device)
        rec_prev = (h_prev, c_prev)

        for _, (X, Y) in enumerate(iter(test_loader)):
            
            Y_hat, c_prev = model(X, rec_prev)
            
            if save_Y_hat:
                Y_hats.append(Y_hat)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    if save_Y_hat:
        return eval_loss, torch.vstack(Y_hats)
    else:
        return eval_loss


def param_grid_search(hidden_dims, n_layerss, **kwargs):
    
    _type = kwargs['_type']
    valid_loader = kwargs['valid_loader']
    test_loader = kwargs['test_loader']
    n_epochs = kwargs['n_epochs']

    final_eval_loss = np.zeros((len(hidden_dims), len(n_layerss)))

    for i, hidden_dim in enumerate(hidden_dims):
        for j, n_layers in enumerate(n_layerss):

            # deep learning model
            if _type == 'FF':
                DNN = FeedForwardDNN(input_dim=16, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=16).to(device)

                # training parameters
                optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
                criterion = nn.MSELoss()    

                train_losses, eval_losses = train(
                    model=DNN,
                    train_loader=valid_loader, test_loader=test_loader,
                    optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
                    verbose=False, force_stop=False)

                print(f'{n_layers} layers, {hidden_dim} hidden units, final eval loss: {eval_losses[-1]}', end='\r')

            elif _type == 'RNN':
                n_linear_layers = n_layers
                DNN = RecurrentDNN(input_dim=16, hidden_dim=hidden_dim, n_linear_layers=n_linear_layers, output_dim=16).to(device)

                optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
                criterion = nn.MSELoss()

                train_losses, eval_losses = train_rnn(
                    model=DNN,
                    train_loader=valid_loader, test_loader=test_loader,
                    optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
                    verbose=False, force_stop=False)

                print(f'{n_linear_layers} linear layers, 1 lstm layer, {hidden_dim} hidden units, final eval loss: {eval_losses[-1]}', end='\r')
            
            final_eval_loss[i,j] = eval_losses[-1]
    
    return final_eval_loss

def repeated_param_grid_search(hidden_dims, n_layerss, n_repeats, **kwargs):

    print('Training on: ', torch.cuda.get_device_name())

    final_eval_loss_matrix = np.zeros((len(hidden_dims), len(n_layerss)))
    start = time.time()
    
    for i in range(n_repeats):
        
        final_eval_loss = param_grid_search(hidden_dims, n_layerss, **kwargs)
        final_eval_loss_matrix += final_eval_loss
        
        repeat_end = time.time()
        elapsed_h, elapsed_m, elapsed_s = elapsed_time(start, repeat_end)
        
        r, c = find_argmin_in_matrix(final_eval_loss)
        print(f'{i+1}th repeat: best hidden unit: {hidden_dims[r]}, best layers: {n_layerss[c]}, final eval loss: {final_eval_loss[r,c]}, time elapsed: {elapsed_h} h {elapsed_m} m {elapsed_s} s')
    
    return final_eval_loss_matrix / n_repeats


def train_varying_data_efficiency(hidden_dim, n_layers, verbose=True, **kwargs):
    final_eval_losses = []
    
    input_dim = kwargs['input_dim']
    output_dim = kwargs['output_dim']
    train_dataset = kwargs['train_dataset']
    test_dataloader = kwargs['test_dataloader']
    n_datas = kwargs['n_datas']
    n_epochs = kwargs['n_epochs']
    batch_size = kwargs['batch_size']
    
    for i, n_data in enumerate(n_datas):
        if verbose:
            print(f'{hidden_dim} hidden dim, {n_layers} layers, using {n_data} out of {max(n_datas)} data points.', end='\r')
            
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset, replacement=False, num_samples=n_data))
        
        DNN = FeedForwardDNN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=output_dim).to(device)
        optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        _, eval_losses = train(
            model=DNN,
            train_loader=train_dataloader, test_loader=test_dataloader,
            optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
            verbose=False, force_stop=False)
        final_eval_losses.append(eval_losses[-1])
        
    return torch.Tensor([final_eval_losses])


def complexity_varying_data_efficiency(hidden_dims, n_layerss, verbose=True, **kwargs):

    final_eval_losses = torch.zeros(len(hidden_dims), len(n_layerss), len(kwargs['n_datas']))
    
    for i, hidden_dim in enumerate(hidden_dims):
        for j, n_layers in enumerate(n_layerss):
            
            if verbose:
                print(f'Running {i * len(hidden_dims) + j + 1} out of {len(hidden_dims) * len(n_layerss)}', end='\r')

            final_eval_losses[i,j,:] = train_varying_data_efficiency(hidden_dim, n_layers, verbose=verbose, **kwargs)
                
    return final_eval_losses


def repeated_complexity_varying_data_efficiency(hidden_dims, n_layerss, n_repeats, verbose=True, **kwargs):
    final_eval_losses = torch.zeros(len(hidden_dims), len(n_layerss), len(kwargs['n_datas']))
    
    start = time.time()
    
    for n in range(n_repeats):
        
        final_eval_losses += complexity_varying_data_efficiency(hidden_dims, n_layerss, verbose=True, **kwargs)
        
        repeat_end = time.time()
        elapsed_h, elapsed_m, elapsed_s = elapsed_time(start, repeat_end)
        
        if verbose:
            print(f'{n+1} of {n_repeats} repeats. Time elapsed: time elapsed: {elapsed_h} h {elapsed_m} m {elapsed_s} s')
        
    return final_eval_losses / n_repeats


def learning_with_plasticity(sigma, alpha=1, transfer_learning=True, **kwargs):

    '''
    Explores how well DNN can learn retrained to approximate a BNN with small plasticity changes.
    Transfer learning is applied using saved DNN weights, on the same training input and new output generated by BNN
    
    returns:
    train_loss:    1D torch.Tensor, history of training losses by epoch
    eval_loss:     1D torch.Tensor, history of evaluation losses by epoch
    '''
    # unpack arguments
    X_train = kwargs['X_train']
    X_test = kwargs['X_test']
    DNN = kwargs['DNN']
    DNN_params = kwargs['DNN_params']
    batch_size = kwargs['batch_size']
    num_epochs = kwargs['num_epochs']
    verbose = True if kwargs['verbose'] == 2 else False
    
    _Y_train, _Y_test = apply_plasticity_and_generate_new_output(sigma, alpha, **kwargs)
    
    if isinstance(DNN, FeedForwardDNN):
        # For feedforward MLP, merge first two dimensions
        X_train = X_train.view(-1, 16)
        _Y_train = _Y_train.view(-1, 16)
        X_test = X_test.view(-1, 16)
        _Y_test = _Y_test.view(-1, 16)
    
    # prepare dataloader
    train_dataset, test_dataset = BNN_Dataset(X_train, _Y_train), BNN_Dataset(X_test, _Y_test)
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 
    
    # load trained DNN
    if transfer_learning:
        DNN.load_state_dict(DNN_params)
    else:
        DNN.reset()
        
    optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    if isinstance(DNN, FeedForwardDNN):
        train_losses, eval_losses, init_eval_loss = train(
            model=DNN, 
            train_loader=train_dataloader, test_loader=test_dataloader,
            optimiser=optimiser, criterion=criterion, num_epochs=num_epochs,
            verbose=verbose, force_stop=False, return_init_eval_loss=True)
        
    if isinstance(DNN, RecurrentDNN):
        train_losses, eval_losses, init_eval_loss = train_rnn(
            model=DNN,
            train_loader=train_dataloader, test_loader=test_dataloader,
            optimiser=optimiser, criterion=criterion, num_epochs=num_epochs,
            verbose=verbose, force_stop=False, return_init_eval_loss=True)
        
    return torch.Tensor(train_losses), torch.Tensor(eval_losses), torch.Tensor([init_eval_loss])


def plasticity_analysis(transfer_learning, **kwargs):
    '''
    Varying plasticity parameter sigma (and alpha), explores how well the trained DNN can approximate the approximate DNN with plasticity changes
    
    :params:
    transfer_learning       whether to learn from trained parameter (True) or retrain from random initialisation (False)
    verbose:                0 or 1 or 2. 0=No log; 1=Only showing progress on which sigma; 2=showing progress for each epoch
    
    returns:
    train_losses_by_sigma:  2D torch.Tensor, training loss history per epoch for each sigma
    eval_losses_by_sigma:   2D torch.Tensor, evaluation loss history per epoch for each sigma
    '''
    # unpack arguments
    sigmas = kwargs['sigmas']
    verbose = kwargs['verbose']
    num_epochs = kwargs['num_epochs']
    v = True if verbose == 2 else False
    
    # store all loss history in tensor
    n_sigmas = sigmas.shape[0]
    train_losses_by_sigma = torch.zeros(n_sigmas, num_epochs)
    eval_losses_by_sigma = torch.zeros(n_sigmas, num_epochs)
    init_eval_losses_by_sigma = torch.zeros(n_sigmas)
    
    for i, sigma in enumerate(sigmas):
        if verbose >= 1:
            print(f'Processing simga={sigma} ({i+1}/{n_sigmas})')
            
        sigma_train_losses, sigma_eval_losses, sigma_init_eval_loss = learning_with_plasticity(
            sigma, alpha=1, 
            transfer_learning=transfer_learning, 
            **kwargs)
        
        train_losses_by_sigma[i,:] = sigma_train_losses
        eval_losses_by_sigma[i,:] = sigma_eval_losses
        init_eval_losses_by_sigma[i] = sigma_init_eval_loss
        
        if verbose >= 1:
            print(f'\t Inital eval loss: {sigma_init_eval_loss}, final eval loss: {sigma_eval_losses[-1]}')
        
    return train_losses_by_sigma, eval_losses_by_sigma, init_eval_losses_by_sigma


def repeated_plasticity_analysis(n_repeats, transfer_learning, **kwargs):
    n_sigmas = kwargs['sigmas'].shape[0]
    num_epochs = kwargs['num_epochs']
    
    train_losses_by_sigma_tensor = torch.zeros(n_repeats, n_sigmas, num_epochs)
    eval_losses_by_sigma_tensor = torch.zeros_like(train_losses_by_sigma_tensor)
    init_losses_by_sigma_tensor = torch.zeros(n_repeats, n_sigmas)
    
    for i in range(n_repeats):
        print(f'Repeat {i+1}/{n_repeats}')
        train_losses_by_sigma, eval_losses_by_sigma, init_eval_losses_by_sigma = plasticity_analysis(transfer_learning=transfer_learning, **kwargs)
        
        train_losses_by_sigma_tensor[i,:] = train_losses_by_sigma
        eval_losses_by_sigma_tensor[i,:] = eval_losses_by_sigma
        init_losses_by_sigma_tensor[i] = init_eval_losses_by_sigma

    means = (torch.mean(train_losses_by_sigma_tensor, 0), torch.mean(eval_losses_by_sigma_tensor, 0), torch.mean(init_losses_by_sigma_tensor, 0))
    stds = (torch.std(train_losses_by_sigma_tensor, 0), torch.std(eval_losses_by_sigma_tensor, 0), torch.std(init_losses_by_sigma_tensor, 0))
        
    return means, stds




if __name__ == '__main__':
    
    X_train, Y_train = load_data('./data/', 'abnn_recur_train_256_0.5_4_0.5.pkl')
    X_test, Y_test = load_data('./data/', 'abnn_recur_test_256_0.5_4_0.5.pkl')
    X_valid, Y_valid = load_data('./data/', 'abnn_recur_valid_256_0.5_4_0.5.pkl')

    batch_size = 200   # number of data points in each mini-batch
    n_train = 10000    # number of data used, from 1 to len(X_train)
    n_epochs = 50      # number of training epochs

    train_dataset = BNN_Dataset(X_train[:n_train], Y_train[:n_train])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = BNN_Dataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = BNN_Dataset(X_valid, Y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # deep learning model
    DNN = RecurrentDNN(input_dim=16, hidden_dim=256, n_linear_layers=1, output_dim=16, n_lstm_layers=1).to(device)

    # training parameters
    optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses, eval_losses = train_rnn(
        model=DNN,
        train_loader=train_dataloader, test_loader=test_dataloader,
        optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
        verbose=True, force_stop=False)
    
    _, Y_test_hat = eval_rnn(DNN, test_dataloader, criterion, save_Y_hat=True)

    plot_loss_curves(train_losses, eval_losses, loss_func='MSE Loss')


