import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from preprocess import train_test_split

if __name__ == '__main__':
    batch_size = 128
    train, test = train_test_split()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    for X, y in train_loader:
        pass
