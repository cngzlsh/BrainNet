import torch
import torch.nn as nn
import numpy as np
from models import *
from utils import *

seed = 1234
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train, Y_train = load_data('./data/', 'train.pkl')
X_test, Y_test = load_data('./data/', 'test.pkl')

total_train = X_train.shape[0]
N_train = 8000
assert N_train < total_train
batch_size = 64

if __name__ == '__main__':
    pass