import torch
import torch.utils.data.dataset as Dataset
from torch.utils.data.dataloader import DataLoader
import pickle as pkl
from public.misc import Vocab

'''
extract useful data from the original dataset
'''


def extract_data(path_r, path_w):
    data = []
    with open(path_r, 'r') as f:
        keys = []
        values = []
        for i, line in enumerate(f):
            if i >= 2:
                if line != '\n':
                    kk = line.split()[0].lower()
                    keys.append(kk)
                    vv = line.split()[3]
                    values.append(vv)
                else:
                    data.append([keys, values])
                    keys = []
                    values = []
    for sent_dict in data:
        values = sent_dict[1]
        for i in range(len(values)):
            now = values[i]
            if now[0] == 'I':
                length = 1
                idx = i
                while i + 1 < len(values) and values[i + 1] == now:
                    length += 1
                    i += 1
                if length == 1:
                    values[idx] = 'S-' + now.split('-')[1]
                else:
                    values[idx] = 'B-' + now.split('-')[1]
                    for j in range(idx + 1, idx + length - 1):
                        values[j] = 'M-' + now.split('-')[1]
                    values[idx + length - 1] = 'E-' + now.split('-')[1]

    with open(path_w, 'w') as f:
        for sent in data:
            for i in range(len(sent[0])):
                f.write(sent[0][i] + ' ' + sent[1][i] + '\n')
            f.write('\n')


def read_data(path):
    sentences = []
    labels = []
    with open(path, 'r') as f:
        sent = []
        label = []
        for line in f:
            if line != '\n':
                sent.append(line.split()[0])
                label.append(line.split()[1])
            else:
                sentences.append(sent)
                labels.append(label)
                sent = []
                label = []
    return sentences, labels


'''
make the data digitalized 
'''


def flatten(tokens):
    return [word for line in tokens for word in line]


class CONLLDataset(Dataset):
    def __init__(self, tokens_sentence, labels, vocab=None, label_token=None):
        self.vocab = Vocab(tokens_sentence, min_freq=5) if vocab is None else vocab
        self.labels = Vocab(labels, has_unk=False) if label_token is None else label_token


if __name__ == '__main__':
    sentences, labels = read_data('./data/train.txt')
    tokens_sentence = flatten(sentences)
    labels = flatten(labels)
    pass
