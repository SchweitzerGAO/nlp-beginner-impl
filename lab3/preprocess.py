import pickle as pkl
import re

import torch
import torch.utils.data.dataset as Dataset
from torch.utils.data.dataloader import DataLoader
from public.misc import Vocab, truncate_pad


def parse2sentence(sentence):
    # remove () in the parsed sentence
    sentence = re.sub('\\(', '', sentence)
    sentence = re.sub('\\)', '', sentence)
    # remove redundant spaces
    sentence = re.sub('\\s{2,}', ' ', sentence)
    return sentence.strip().lower()


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
            if test and i > 1000:
                break
    return premises, hypotheses, labels


def tokenize_word(lines):
    if isinstance(lines, list):
        return [line.split() for line in lines]
    else:
        return lines.lower().split()


class SNLIDataset(Dataset.Dataset):
    def __init__(self, premises, hypotheses, labels, length, vocab=None, trunc_pad=True):
        self.length = length
        premise_tokens = tokenize_word(premises)
        hypotheses_tokens = tokenize_word(hypotheses)
        self.vocab = Vocab(premise_tokens + hypotheses_tokens, min_freq=5,
                           reserved_tokens=['<pad>']) if vocab is None else vocab
        self.premises = self._pad(premise_tokens) if trunc_pad else self._direct(premise_tokens)
        self.hypotheses = self._pad(hypotheses_tokens) if trunc_pad else self._direct(hypotheses_tokens)
        self.labels = torch.LongTensor(labels)

    def _pad(self, tokens):
        return [truncate_pad(self.vocab[token], self.length, self.vocab['<pad>']) for token in tokens]

    def _direct(self, tokens):
        return [torch.tensor(self.vocab[token]) for token in tokens]

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]


def load_train_data(batch_size=32, length=50, test=False, trunc_pad=True, num_workers=0):
    premises, hypotheses, labels = read_data('./snli_1.0/snli_1.0_train.txt', test=test)
    with open('./train_vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    train_set = SNLIDataset(premises, hypotheses, labels, length=length, vocab=vocab, trunc_pad=trunc_pad)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers), vocab


if __name__ == '__main__':
    premises, hypotheses, labels = read_data('./snli_1.0/snli_1.0_train.txt')
    with open('./train_vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    train_set = SNLIDataset(premises, hypotheses, labels, length=50, vocab=vocab)

    for (A, B), y in train_set:
        pass
