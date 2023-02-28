from public.misc import Vocab, truncate_pad
import re

import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torch


def parse2sentence(sentence):
    # remove () in the parsed sentence
    sentence = re.sub('\\(', '', sentence)
    sentence = re.sub('\\)', '', sentence)
    sentence = re.sub('\\s{2,}', ' ', sentence)
    return sentence.strip()


def read_data(path, test=False):
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    premises = []
    hypotheses = []
    labels = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 1:
                data = line.split('\t')
                label = data[0]
                if label in label_set:
                    premises.append(parse2sentence(data[1]))
                    hypotheses.append(parse2sentence(data[2]))
                    labels.append(label_set[data[0]])
            if i >= 1000 and test:
                break
    return premises, hypotheses, labels


def tokenize_word(lines):
    return [line.split() for line in lines]


class SNLIDataset(Dataset.Dataset):
    def __init__(self, premises, hypotheses, labels, length, vocab=None):
        self.length = length
        premise_tokens = tokenize_word(premises)
        hypotheses_tokens = tokenize_word(hypotheses)
        self.vocab = Vocab(premise_tokens + hypotheses_tokens, min_freq=5,
                           reserved_tokens=['<pad>']) if vocab is None else vocab
        self.premises = self.pad(premise_tokens)
        self.hypotheses = self.pad(hypotheses_tokens)
        self.labels = torch.Tensor(labels)

    def pad(self, tokens):
        return [truncate_pad(self.vocab[token], self.length, self.vocab['<pad>']) for token in tokens]

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]


if __name__ == '__main__':
    premises, hypotheses, labels = read_data('./snli_1.0/snli_1.0_train.txt', test=True)
    train_set = SNLIDataset(premises, hypotheses, labels, length=50)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    for (P, H), y in train_loader:
        pass
