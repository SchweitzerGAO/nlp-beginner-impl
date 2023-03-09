import random
import re
from public.misc import *
import torch
from torch import nn


def read_data():
    with open('data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub('[^\u4e00-\u9fa5]+', '', line).strip().lower() for line in lines]


def tokenize_char(lines):
    return [list(line) for line in lines]


def load_corpus(max_tokens=None):
    lines = read_data()
    tokens = tokenize_char(lines)
    vocab = Vocab(tokens=tokens)
    corpus = [vocab[token] for line in tokens for token in line]  # get the 'digitalized' character
    if max_tokens is not None:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# randomly loaded data
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)  # here shuffles the indices

    def get_data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [get_data(j) for j in initial_indices_per_batch]
        y = [get_data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(y)


# sequentially loaded data
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  # a preparation of reshape
    # the batches not shuffled thus it is sequential
    X_batch = torch.tensor(corpus[offset: offset + num_tokens])
    y_batch = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    X_batch, y_batch = X_batch.reshape(batch_size, -1), y_batch.reshape(batch_size, -1)
    num_batches = X_batch.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = X_batch[:, i: i + num_steps]
        y = y_batch[:, i: i + num_steps]
        yield X, y


class PoemDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_poems(batch_size, num_steps, use_random=False, max_tokens=None):
    data = PoemDataLoader(batch_size, num_steps, use_random, max_tokens)
    return data, data.vocab


if __name__ == '__main__':
    data, voc = load_poems(batch_size=64, num_steps=10)
    embed = nn.Embedding(len(voc), 100)
    for X, y in data:
        X_embed = embed(X)
        pass
