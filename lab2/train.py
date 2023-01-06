import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from preprocess import TextSentimentDataset, train_test_split, MAX_SENTENCE_SIZE
from model import TextCNN

vec_dim = 5
batch_size = 128
dataset = TextSentimentDataset('../lab1/data/train.tsv', './word_vectors/random.pkl', vec_dim)
train_set, test_set = train_test_split(dataset)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = TextCNN(vec_dim, dataset.num_cls)
optimizer = optim.Adam(params=model.parameters(), lr=0.5)
loss_function = nn.CrossEntropyLoss()
has_cuda = torch.cuda.is_available()


def accuracy(pred, gt):
    pred = pred.argmax(dim=1)
    diff = (pred - gt).numpy()
    return np.count_nonzero(diff == 0) / len(diff)


def train_epoch(net, loss_func, opt):
    acc = []
    losses = []
    net.train()
    for X, y in train_loader:
        if has_cuda:
            X = X.cuda()
            y = y.cuda()
        pred = net(X)
        acc.append(accuracy(pred, y))
        loss = loss_func(pred, y)
        losses.append(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    avg_acc = np.array(acc).mean()
    avg_loss = np.array(losses).mean()
    return avg_acc, avg_loss


def evaluate(net):
    acc = []
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            if has_cuda:
                X = X.cuda()
                y = y.cuda()
            pred = net(X)
            acc.append(accuracy(pred, y))
    avg = np.array(acc).mean()
    return avg


def train(epoch,save_path):
    for i in range(epoch):
        acc_train, loss_train = train_epoch(model, loss_function, optimizer)
        acc_test = evaluate(model)
        print(
            f'Epoch({i + 1}/{epoch}): '
            f'loss:{round(loss_train, 4)}; '
            f'train_acc:{round(acc_train * 100., 4)} %; '
            f'test_acc:{round(acc_test * 100., 4)} %')
        if (i+1) % 10 == 0:
            torch.save(model.state_dict(),save_path+f'/{i+1}_{batch_size}.pt')


if __name__ == '__main__':
    if has_cuda:
        model = model.cuda()
    ep = 20
    train(ep,'./models/textcnn_random')
