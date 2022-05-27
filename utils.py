import pickle
import torch
import torch.nn as nn
import time
import os

def save_data(X, Y, path, filename):

    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + filename, 'wb') as f:
        pickle.dump((X, Y), f)
    
    f.close()


def load_data(path, filename):

    with open(path + filename, 'rb') as f:
        X, Y = pickle.load(f)
    f.close()

    return X, Y


class MLP(nn.Module):
    '''
    Deep neural network used to learn the firing pattern of biological networks
    '''
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x):
        pass


def elapsed_time(start, end):
    secs = end - start
    mins = int(secs / 60)
    hrs = int(mins / 60)
    return hrs, mins % 60, secs % 60


def train(model, train_loader, optimiser, criterion, num_epochs, verbose=True, stop=False):
    
    model.train()
    start = time.time()

    for epoch in range(num_epochs):
        
        epoch_loss = 0
        
        for i, (X, Y) in enumerate(iter(train_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

            if stop and i == 20:
                break
        
        if verbose:
            epoch_end = time.time()
            hrs, mins, secs = elapsed_time(start, epoch_end)
            print(f'Epoch {epoch+1} completed with loss {int(epoch_loss)}. Time elapsed: {hrs} h {mins} m {secs} s.')
    
    end = time.time()
    hrs, mins, secs = elapsed_time(start, time.time())

    print(f'Training completed. Time elapsed: {hrs} h {mins} m {secs} s.')


def eval(model, eval_loader, criterion):
    model.eval()

    with torch.no_grad():
        for i, (X, Y) in iter(eval_loader):
            pass

