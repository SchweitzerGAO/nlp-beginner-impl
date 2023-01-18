import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from preprocess import TextSentimentDataset, train_test_split


class TextCNN(nn.Module):
    def __init__(self, vec_dim, out_channel, filter_num=100, kernels=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filter_num, (kernel, vec_dim)),
            )
            for kernel in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filter_num * len(kernels), 128)
        self.fc2 = nn.Linear(128, out_channel)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = [F.leaky_relu(conv(x)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in out]
        out = torch.cat(out, dim=1)
        out = self.dropout(out)
        out = self.fc2(self.fc1(out))
        return out


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mode='gru', num_layers=1, dropout=0.5, bi_dir=False):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        if num_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0
        if mode == 'gru':
            self.rnn_layer = nn.GRU(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    dropout=self.dropout,
                                    bidirectional=bi_dir)
        elif mode == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=self.dropout,
                                     bidirectional=bi_dir)
        if not bi_dir:
            self.num_dir = 1
        else:
            self.num_dir = 2
        self.fc = nn.Linear(hidden_size * self.num_dir, output_size)

    def init_state(self, device, batch_size=1):
        if not isinstance(self.rnn_layer, nn.LSTM):
            # nn.GRU uses Tensor as a hidden state
            return torch.zeros((self.num_dir * self.rnn_layer.num_layers,
                                batch_size, self.hidden_size), device=device)
        else:
            # nn.LSTM uses tuple as a hidden state
            return (torch.zeros((
                self.num_dir * self.rnn_layer.num_layers,
                batch_size, self.hidden_size)).to(device),
                    torch.zeros((
                        self.num_dir * self.rnn_layer.num_layers,
                        batch_size, self.hidden_size), device=device))

    def forward(self, x, state):
        out, state = self.rnn_layer(x, state)
        out = self.fc(out)
        out = out[:, -1, :]  # get the last sequence of prediction in classification task
        return out, state


if __name__ == '__main__':
    vec_dim = 5
    dataset = TextSentimentDataset('../lab1/data/train.tsv', './word_vectors/random.pkl', vec_dim)
    net = TextRNN(vec_dim, 256, dataset.num_cls,bi_dir=True)
    state = net.init_state(batch_size=64, device='cpu')
    train_set, test_set = train_test_split(dataset)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    net.train()
    for X, y in train_loader:
        out, state = net(X, state)
        pass
