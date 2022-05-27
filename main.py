import torch
import torch.nn as nn
import numpy as np
from utils import *

X_train, Y_train = load_data('./data/', 'train.pkl')

class DNN(nn.Module):
    def __init__(self):
        pass