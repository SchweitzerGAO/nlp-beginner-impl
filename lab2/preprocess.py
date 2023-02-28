import pickle as pkl

import pandas as pd
import torch
import torch.utils.data.dataset as Dataset
from torch import nn
from torch.utils.data import random_split

MAX_SENTENCE_SIZE = 40


def word_embed_random():
    df = pd.read_csv('../lab1/data/train.tsv', sep='\t')
    corpus = df['Phrase']
    res = dict()
    for sent in corpus:
        sent_set = set(sent.lower().split())
        embed = nn.Embedding(len(sent_set), 5)
        idx = {word: i for i, word in enumerate(sent_set)}
        for word in sent_set:
            wv = embed(torch.LongTensor([idx[word]]))
            if res.get(word, None) is None:
                res[word] = wv.reshape(5)
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
    with open('../public/glove_6B_100d.pkl', 'wb') as f:
        pkl.dump(res, f)


def sentence_to_vector(sentence, vec_dict, dim):
    max_length = MAX_SENTENCE_SIZE
    res = torch.empty([max_length, dim])
    words = sentence.lower().split()
    while len(words) < max_length:
        words.append('<pad>')
    for i, word in enumerate(words[:max_length]):
        vec = vec_dict.get(word, None)
        if vec is None:
            res[i] = torch.randn([1, dim])
        else:
            res[i] = vec
    return res


class TextSentimentDataset(Dataset.Dataset):
    def __init__(self, data_path, vec_path, dim):
        df = pd.read_csv(data_path, sep='\t')
        corpus = df['Phrase']
        with open(vec_path, 'rb') as f:
            dic = pkl.load(f)
        self.dic = dic
        self.dim = dim
        self.data = corpus
        self.label = df['Sentiment']
        self.num_cls = len(set(self.label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = sentence_to_vector(self.data[idx], self.dic, self.dim)
        label = self.label[idx]
        return data, label


def train_test_split(dataset, ratio=0.8):
    length = len(dataset)
    train_length = int(length * ratio)
    test_length = length - train_length
    train_set, test_set = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(0))
    return train_set, test_set


if __name__ == '__main__':
    '''
    word-embedding
    '''
    # word_embed_random()

    # with open('./word_vectors/glove_6B_50d.pkl', 'rb') as f:
    #     dic = pkl.load(f)
    # sentence = 'is also good for the gander , some of which occasionally amuses but none of which amounts to much of ' \
    #            'a story . '
    # vec = sentence_to_vector(sentence, dic, 50)
    # print(vec)
