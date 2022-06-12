from re import A
import torch
import torch.nn as nn
import numpy as np
from models import *
from utils import *
from torch.utils.data import DataLoader

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data generated from approximate BNN
X_train, Y_train = load_data('./data/', 'train_abnn_16_256_0.5_4_16_0.5_10000.pkl')
X_test, Y_test = load_data('./data/', 'test_abnn_16_256_0.5_4_16_0.5_1000.pkl')

batch_size = 50    # number of data points in each mini-batch
n_train = 10000    # number of data used, from 1 to len(X_train)
n_epochs = 25      # number of training epochs


def param_grid_search(hidden_dims = [2, 4, 8, 16, 32, 64, 128, 256], n_layerss = [1, 2, 3, 4, 5, 6, 7, 8]):
    final_eval_losses =[]

    for hidden_dim in hidden_dims:
        for n_layers in n_layerss:

            # deep learning model
            DNN = FeedForwardDNN(input_dim=16, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=16).to(device)

            # training parameters
            optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
            criterion = nn.MSELoss()    

            # visualise_prediction(Y_test[v_idx,:], DNN(X_test[v_idx,:]))

            train_losses, eval_losses = train(
                model=DNN,
                train_loader=train_dataloader, test_loader=test_dataloader,
                optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
                verbose=False, force_stop=False)
            
            # plot_loss_curves(train_losses, eval_losses, loss_func='MSE Loss')

            # visualise_prediction(Y_test[v_idx,:], DNN(X_test[v_idx,:]))
            print(f'{n_layers} layers, {hidden_dim} hidden units, final eval loss: {eval_losses[-1]}')
            final_eval_losses.append(eval_losses[-1])
    
    return final_eval_losses


if __name__ == '__main__':

    v_idx = np.random.randint(0, len(Y_test)) # a random label in the test set to visualise

    train_dataset = BNN_Dataset(X_train[:n_train], Y_train[:n_train])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = BNN_Dataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # deep learning model
    DNN = FeedForwardDNN(input_dim=16, hidden_dim=128, n_layers=2, output_dim=16).to(device)

    # training parameters
    optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
    criterion = nn.MSELoss()    

    visualise_prediction(Y_test[v_idx,:], DNN(X_test[v_idx,:]))

    train_losses, eval_losses = train(
        model=DNN,
        train_loader=train_dataloader, test_loader=test_dataloader,
        optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
        verbose=True, force_stop=False)
    
    # plot_loss_curves(train_losses, eval_losses, loss_func='MSE Loss')

    visualise_prediction(Y_test[v_idx,:], DNN(X_test[v_idx,:]))

    # predicting the mean
    mean = torch.mean(Y_test)
    assert False

