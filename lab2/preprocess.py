import pandas as pd
import pickle as pkl
import torch
from torch import nn


def read_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df['Phrase'], df['Sentiment']


def word_embed_random():
    corpus, label = read_dataset('../lab1/data/train.tsv')
    res = dict()
    for sent in corpus:
        sent_set = set(sent.lower().split(' '))
        embed = nn.Embedding(len(sent_set), 5)
        idx = {word: i for i, word in enumerate(sent_set)}
        for word in sent_set:
            wv = embed(torch.LongTensor([idx[word]]))
            if res.get(word, None) is None:
                res[word] = wv
        print(sent)
    with open('./word_vectors/random.pkl', 'wb') as f:
        pkl.dump(res, f)


def word_embed_glove():
    res = dict()
    with open('./glove_6B/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split(' ')
            vec = torch.Tensor([float(number) for number in line[1:]])
            res[line[0]] = vec
    with open('./word_vectors/glove_6B_100d.pkl', 'wb') as f:
        pkl.dump(res, f)


if __name__ == '__main__':
    with open('./word_vectors/glove_6B_100d.pkl', 'rb') as f:
        res = pkl.load(f)
        pass
