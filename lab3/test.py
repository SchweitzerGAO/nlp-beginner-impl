import torch
import torch.nn.functional as F
from model import ESIM
from preprocess import read_data, SNLIDataset
from torch.utils.data.dataloader import DataLoader
import pickle as pkl
import numpy as np


def accuracy(pred, gt):
    pred = F.softmax(pred, dim=1)
    pred = torch.max(pred, dim=1)[1]
    return (pred == gt).sum() / len(gt)


batch_size = 1
embed_size = 100
length = 50
hidden_size_lstm = 128
hidden_size_dense = 128

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_test_data():
    premises, hypotheses, labels = read_data('./snli_1.0/snli_1.0_test.txt')
    with open('./train_vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    test_set = SNLIDataset(premises, hypotheses, labels, length=length, vocab=vocab)

    return DataLoader(test_set, batch_size, shuffle=False), vocab


def test(weight_path):
    test_loader, vocab = load_test_data()
    net = ESIM(vocab, embed_size, length, hidden_size_lstm, hidden_size_dense, output_size=3)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()
    acc = []
    with torch.no_grad():
        for (A, B), y in test_loader:
            A = A.to(device)
            B = B.to(device)
            y = y.to(device)
            y_hat = net(A, B)
            acc.append(accuracy(y_hat, y).cpu())
        avg_acc = np.array(acc).mean()
    print(f'{round(avg_acc, 4) * 100.} %')


if __name__ == '__main__':
    test('./saved_models/10_256.pt')
