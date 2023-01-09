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

    def forward(self, x):
        x = x.unsqueeze(1)
        out = [F.leaky_relu(conv(x)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, dim=1)
        # out = out.view(x.size(0), -1)
        out = self.dropout(out)
        out = self.fc2(self.fc1(out))
        return out

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)



if __name__ == '__main__':
    vec_dim = 50
    dataset = TextSentimentDataset('../lab1/data/train.tsv', './word_vectors/glove_6B_50d.pkl', vec_dim)
    net = TextCNN(dataset.dim, dataset.num_cls,kernels=tuple([3]))
    net.init()
    train_set, test_set = train_test_split(dataset)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    net.train()
    for X, y in train_loader:
        out = net(X)
        pass
