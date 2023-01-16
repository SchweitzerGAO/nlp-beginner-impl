import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import TextCNN, TextRNN
from preprocess import TextSentimentDataset, train_test_split

from matplotlib import pyplot as plt

'''
configurations
'''
# hyper-params
lr = 1e-3
vec_dim = 5
batch_size = 64
hidden_size = 256

# loss function & optimizer
loss_function = nn.CrossEntropyLoss()

# GPU accessibility
has_cuda = torch.cuda.is_available()

# plot
train_acc = []
test_acc = []

# dataset
dataset = TextSentimentDataset('../lab1/data/train.tsv', './word_vectors/random.pkl', vec_dim)
train_set, test_set = train_test_split(dataset)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,drop_last=True) # drop_last = True only when training RNN model
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,drop_last=True)

'''
accuracy
'''


def accuracy(pred, gt):
    pred = torch.max(pred, dim=1)[1]
    return (pred == gt).sum() / len(gt)


'''
gradient clipping
'''


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    # L2-norm
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # norm too big, then decrease this
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


'''
train with CNN
'''
# TextCNN
model_cnn = TextCNN(vec_dim, dataset.num_cls)
model_cnn.init_weight()
optimizer_cnn = optim.Adam(params=model_cnn.parameters(), lr=lr, weight_decay=1e-3)


# TextCNN train and evaluate
def train_epoch_cnn(net, loss_func, opt):
    acc = []
    losses = []
    net.train()
    for X, y in train_loader:
        if has_cuda:
            X = X.cuda()
            y = y.cuda()
        pred = net(X)
        acc.append(accuracy(pred, y).cpu())
        loss = loss_func(pred, y)
        losses.append(loss.cpu().detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    avg_acc = np.array(acc).mean()
    avg_loss = np.array(losses).mean()
    return avg_acc, avg_loss


def evaluate_cnn(net):
    acc = []
    net.eval()
    with torch.no_grad():
        for X, y in test_loader:
            if has_cuda:
                X = X.cuda()
                y = y.cuda()
            pred = net(X)
            acc.append(accuracy(pred, y).cpu())
    avg = np.array(acc).mean()
    return avg


def train_cnn(net, loss_func, opt, epoch, save_path):
    if has_cuda:
        net = net.cuda()
    for i in range(epoch):
        acc_train, loss_train = train_epoch_cnn(net, loss_func, opt)
        acc_test = evaluate_cnn(net)
        print(
            f'Epoch({i + 1}/{epoch}): '
            f'loss:{round(loss_train, 4)}; '
            f'train_acc:{round(acc_train * 100., 4)} %; '
            f'test_acc:{round(acc_test * 100., 4)} %')
        train_acc.append(acc_train)
        test_acc.append(acc_test)
        if (i + 1) % 10 == 0:
            torch.save(net.state_dict(), save_path + f'/{i + 1}_{batch_size}.pt')
            plot(f'./plots/cnn_{i + 1}.png')


'''
train with RNN
'''

model_rnn = TextRNN(vec_dim, hidden_size, dataset.num_cls)
optimizer_rnn = optim.Adam(params=model_rnn.parameters(), lr=lr)


# TextRNN train and evaluate
def train_epoch_rnn(net, loss_func, opt, device, state=None, grad_clip=False):
    acc = []
    losses = []
    net.train()
    for X, y in train_loader:
        if state is None:
            state = net.init_state(batch_size=batch_size, device=device)
        else:
            if not isinstance(state, tuple):
                state.detach_()  # detach the computation graph
            else:
                for s in state:
                    s.detach_()
        if has_cuda:
            X = X.cuda()
            y = y.cuda()
        pred, state = net(X, state)
        acc.append(accuracy(pred, y).cpu())
        loss = loss_func(pred, y)
        losses.append(loss.cpu().detach())
        opt.zero_grad()
        loss.backward()
        if grad_clip:
            grad_clipping(net, 1)
        opt.step()
    avg_acc = np.array(acc).mean()
    avg_loss = np.array(losses).mean()
    return avg_acc, avg_loss


def evaluate_rnn(net, device):
    acc = []
    net.eval()
    with torch.no_grad():
        state = net.init_state(batch_size=batch_size, device=device)
        for X, y in test_loader:
            if has_cuda:
                X = X.cuda()
                y = y.cuda()
            pred, state = net(X, state)
            acc.append(accuracy(pred, y).cpu())
    avg = np.array(acc).mean()
    return avg


def train_rnn(net, loss_func, opt, epoch, save_path, grad_clip=False):
    device = 'cpu'
    if has_cuda:
        net = net.cuda()
        device = 'cuda:0'
    for i in range(epoch):
        acc_train, loss_train = train_epoch_rnn(model_rnn, loss_func, opt, device=device, grad_clip=grad_clip)
        acc_test = evaluate_rnn(net,device)
        print(
            f'Epoch({i + 1}/{epoch}): '
            f'loss:{round(loss_train, 4)}; '
            f'train_acc:{round(acc_train * 100., 4)} %; '
            f'test_acc:{round(acc_test * 100., 4)} %')
        train_acc.append(acc_train)
        if (i + 1) % 10 == 0:
            torch.save(net.state_dict(), save_path + f'/{i + 1}_{batch_size}.pt')
            plot(f'./plots/rnn_{i + 1}.png')


'''
plot
'''


def plot(file_name):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.ylim(ymin=0.48, ymax=0.65)
    plt.title('Accuracy')
    plt.legend(["train", 'test'])
    plt.savefig(file_name)


if __name__ == '__main__':
    ep = 50
    train_rnn(model_rnn, loss_function, optimizer_rnn, ep, './saved_models/textrnn_glove50')
