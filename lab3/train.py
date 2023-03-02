import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from preprocess import load_train_data
from matplotlib import pyplot as plt
from model import ESIM
import numpy as np

'''
hyper-parameters
'''
batch_size = 32
embed_size = 100
length = 50
hidden_size_lstm = 128
hidden_size_dense = 128
momentum = 0.9
lr = 4e-4
ep = 100
dropout = 0.5

'''
plots
'''
loss = []
accuracy = []

# gpu accessibility
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# preparation
train_loader, vocab = load_train_data(batch_size=batch_size, length=length)
net = ESIM(vocab, embed_size, length, hidden_size_lstm, hidden_size_dense, output_size=3, dropout=0.5)
net = net.to(device)
loss_function = nn.CrossEntropyLoss()  # has a softmax layer embedded
optimizer = optim.Adam(net.parameters(), lr=lr)


def plot(file_name, to_plot, title):
    plt.plot(to_plot)
    plt.title(title)
    plt.savefig(file_name)

def accuracy(pred, gt):
    pred = F.softmax(pred, dim=1)
    pred = torch.max(pred, dim=1)[1]
    return (pred == gt).sum() / len(gt)


def train_epoch():
    acc = []
    losses = []
    for (A, B), y in train_loader:
        optimizer.zero_grad()
        A = A.to(device)
        B = B.to(device)
        y = y.to(device)
        y_hat = net(A, B)
        acc.append(accuracy(y_hat, y).cpu())
        loss = loss_function(y_hat, y.long())
        losses.append(loss.cpu().detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(acc).mean()
    avg_loss = np.array(losses).mean()
    return avg_acc, avg_loss


def train(save_path):
    net.train()
    for i in range(ep):
        l, acc = train_epoch()
        print(
            f'Epoch({i + 1}/{ep}): '
            f'loss:{round(l, 4)}; '
            f'accuracy:{round(acc * 100., 4)} %')
        loss.append(l)
        accuracy.append(acc)
        if (i + 1) % 10 == 0:
            torch.save(net.state_dict(), save_path + f'/{i + 1}_{batch_size}.pt')
    plot(f'./plots/{ep}_loss.png', loss, 'Loss')
    plot(f'./plots/{ep}_loss.png', accuracy, 'Accuracy')


if __name__ == '__main__':
    train('./saved_models')
