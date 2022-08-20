import pandas as pd
import numpy as np


class BOW:
    def __init__(self, train_path):
        df = pd.read_csv(train_path, sep='\t')
        self.corpus = df['Phrase']
        self.label = df['Sentiment']
        self.vocab = []
        self.data = []
        self.idx = dict()
        for phrase in self.corpus:
            self.vocab = list(set(self.vocab + phrase.lower().split(' ')))
        for i, voc in enumerate(self.vocab):
            self.idx[voc] = i

    def __getitem__(self, item):
        return self.vocab[item]

    def voc2idx(self, voc):
        return self.idx.get(voc, -1)

    def generate_bag(self, phrase):
        bag = np.array([0] * len(self.vocab), dtype=np.float64)
        words = phrase.split()
        for word in words:
            word_idx = self.voc2idx(word)
            if word_idx != -1 and bag[word_idx] == 0:
                bag[word_idx] = float(words.count(word))
        return bag

    def init_data(self):
        for phrase in self.corpus:
            self.data.append(self.generate_bag(phrase))

def dataloader_bow(batch_size,shuffle=True):
    pass