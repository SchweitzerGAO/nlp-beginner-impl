import pandas as pd
import numpy as np


class BOW:
    def __init__(self, train_path):
        df = pd.read_csv(train_path, sep='\t')
        self.corpus = df['Phrase']
        self.label = df['Sentiment']
        self.vocab = []  # index-vocabulary
        self.idx = dict()  # vocabulary-index
        for phrase in self.corpus:
            phrase = phrase.lower()
            self.vocab = list(set(self.vocab + phrase.split(' ')))
        for i, voc in enumerate(self.vocab):
            self.idx[voc] = i
        # delete null character
        self.vocab.remove('')
        self.idx.pop('')

    def __getitem__(self, item):
        return self.vocab[item]

    def voc2idx(self, voc):
        return self.idx.get(voc, -1)

    def generate_bag(self, phrase):
        bag = np.array([0.] * len(self.vocab), dtype=np.float32)
        words = phrase.split()
        for word in words:
            word_idx = self.voc2idx(word)
            if word_idx != -1 and bag[word_idx] == 0:
                bag[word_idx] = float(words.count(word))
        return bag


def train_test_split(bow, test_ratio=0.2):
    len_all = len(bow.corpus)
    len_train = int(len_all * (1 - test_ratio))
    train_set = bow.corpus[:len_train]
    test_set = bow.corpus[len_train + 1:]
    return train_set, test_set


def dataloader_bow(train_set, batch_size, shuffle=True):
    len_train = len(train_set)
    num_batch = len_train // batch_size
    for i in range(0, num_batch * batch_size, batch_size):
        X = np.array([bow.generate_bag(phrase) for phrase in bow.corpus[i:i + batch_size]])
        y = np.array(bow.label[i:i + batch_size])
        if shuffle:
            concat = np.concatenate((X, y.reshape(batch_size, -1)), dim=1)
            np.random.shuffle(concat)
            X = concat[:, 0:-1]
            y = concat[:, -1]
        yield X, y


if __name__ == '__main__':
    bow = BOW('./data/train.tsv')
    train_set, test_set = train_test_split(bow)
    print(len(train_set))
    for X, y in dataloader_bow(train_set, 32):
        print(X.shape, y.shape)
        break
