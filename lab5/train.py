import math
import torch
from torch import nn, optim
from model import TextGenerator
from preprocess import load_poems
from matplotlib import pyplot as plt
from public.misc import grad_clipping

'''
hyper-parameters
'''
batch_size = 64
num_steps = 14
embedding_size = 100
hidden_size = 256
lr = 1e-3

# GPU accessibility
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# plot perplexity
ppl = []

data, voc = load_poems(batch_size, num_steps)
net = TextGenerator(embedding_size, hidden_size, len(voc))
loss_function = nn.CrossEntropyLoss()  # has a softmax layer embedded
optimizer = optim.Adam(net.parameters(), lr=lr)

net = net.to(device)


def perplexity(loss):
    return math.exp(loss)


def train_epoch():
    state = net.begin_state(batch_size=batch_size, device=device)
    num_batch = 0
    perp = 0
    for X, y in data:
        if isinstance(net, nn.Module) and not isinstance(state, tuple):
            state.detach_()
        else:
            for s in state:
                s.detach_()
        num_batch += 1
        X = X.to(device)
        y = y.to(device)
        y_hat, state = net(X, state)
        loss = loss_function(y_hat, y).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_clipping(net, 1)
        optimizer.step()
        perp += perplexity(loss)
    return perp / num_batch


def train(epoch):
    net.train()
    for i in range(epoch):
        perp = train_epoch()
        print(
            f'Epoch({i + 1}/{epoch}): '
            f'perplexity:{round(perp, 4)}')
        ppl.append(perp)


def plot(file_name, ep):
    plt.plot(ppl)
    plt.ylim(ymin=0, ymax=10.0)
    plt.xlim(xmin=15, xmax=ep)
    plt.title('Perplexity')
    plt.legend(['train'])
    plt.savefig(file_name)


if __name__ == '__main__':
    ep = 100
    save_path = './saved_model/gru'
    train(ep)
    plot(f'./plots/{ep}_gru_{num_steps}.png', ep)
    torch.save(net.state_dict(), save_path + f'/{ep}_{batch_size}_{num_steps}.pt')
