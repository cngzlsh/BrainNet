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
X_train, Y_train = load_data('./data/', 'train.pkl')
X_test, Y_test = load_data('./data/', 'test.pkl')

batch_size = 50    # number of data points in each mini-batch
n_train = 10000    # number of data used, from 1 to len(X_train)
n_epochs = 20      # number of training epochs

# deep learning model
DNN = FeedForwardDNN1(input_dim=16, hidden_dim=64, n_layers=4, output_dim=16).to(device)

# training parameters
optimiser = torch.optim.Adam(DNN.parameters(), lr=1e-3)
criterion = nn.MSELoss()


if __name__ == '__main__':

    v_idx = np.random.randint(0, len(Y_test)) # a random label in the test set to visualise

    train_dataset = BNN_Dataset(X_train[:n_train], Y_train[:n_train])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = BNN_Dataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    visualise_prediction(Y_test[v_idx,:], DNN(X_test[v_idx,:]))

    train_losses, eval_losses = train(
        model=DNN,
        train_loader=train_dataloader, test_loader=test_dataloader,
        optimiser=optimiser, criterion=criterion, num_epochs=n_epochs,
        verbose=True, force_stop=False)
    
    plot_loss_curves(train_losses, eval_losses, loss_func='MSE Loss')

    visualise_prediction(Y_test[v_idx,:], DNN(X_test[v_idx,:]))