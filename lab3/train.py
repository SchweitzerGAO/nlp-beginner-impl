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
batch_size = 256
embed_size = 100
length = 50
hidden_size_lstm = 128
hidden_size_dense = 128
lr = 4e-4
ep = 20
dropout = 0.5

'''
plots
'''
loss = []
accuracies = []

# gpu accessibility
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# preparation
train_loader, vocab = load_train_data(batch_size=batch_size, length=length, num_workers=8)
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
        A = A.to(device)
        B = B.to(device)
        y = y.to(device)
        y_hat = net(A, B)
        acc.append(accuracy(y_hat, y).cpu())
        l = loss_function(y_hat, y)
        losses.append(l.cpu().detach())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    avg_acc = np.array(acc).mean()
    avg_loss = np.array(losses).mean()
    return avg_acc, avg_loss


def train(save_path):
    net.train()
    for i in range(ep):
        acc, l = train_epoch()
        print(
            f'Epoch({i + 1}/{ep}): '
            f'loss:{round(l, 4)}; '
            f'accuracy:{round(acc * 100., 4)} %')
        loss.append(l)
        accuracies.append(acc)
        if (i + 1) % 10 == 0:
            torch.save(net.state_dict(), save_path + f'/{i + 1}_{batch_size}.pt')
    plot(f'./plots/{ep}_loss.png', loss, 'Loss')
    plot(f'./plots/{ep}_loss.png', accuracy, 'Accuracy')


if __name__ == '__main__':
    train('./saved_models')
