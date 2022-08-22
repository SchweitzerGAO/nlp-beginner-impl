import pandas as pd
import numpy as np
import pickle as pkl


class BOW:
    def __init__(self, train_path, data_path='./proceeded_data/data.pkl', mode='r'):
        df = pd.read_csv(train_path, sep='\t')
        self.corpus = df['Phrase']
        self.cls = df['Sentiment']
        self.vocab = []  # index-vocabulary
        self.num_cls = len(set(self.cls))
        self.idx = dict()  # vocabulary-index
        if mode == 'w':
            for phrase in self.corpus:
                phrase = phrase.lower()
                self.vocab = list(set(self.vocab + phrase.split(' ')))
            for i, voc in enumerate(self.vocab):
                self.idx[voc] = i
            # delete null character
            self.vocab.remove('')
            self.idx.pop('')
            # save data to .pkl file
            data = dict()
            data['vocab'] = self.vocab
            data['idx'] = self.idx
            with open(data_path, 'wb') as wf:
                pkl.dump(data, wf)
        elif mode == 'r':
            with open(data_path, 'rb') as rf:
                data = pkl.load(rf)
            self.vocab = data['vocab']
            self.idx = data['idx']

    def __getitem__(self, idx):
        return self.vocab[idx]

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


def dataloader_bow(bow, train_set, batch_size, shuffle=True):
    len_train = len(train_set)
    num_batch = len_train // batch_size
    for i in range(0, num_batch * batch_size, batch_size):
        X = np.array([bow.generate_bag(phrase) for phrase in bow.corpus[i:i + batch_size]])
        labels = np.array(bow.cls[i:i + batch_size])
        if shuffle:
            concat = np.concatenate((X, labels.reshape(batch_size, -1)), axis=1)
            np.random.shuffle(concat)
            X = concat[:, 0:-1]
            labels = concat[:, -1]
        y = list(map(lambda x: int(x - 1), labels))
        y = np.eye(bow.num_cls, dtype=np.float32)[y]
        yield X, y


if __name__ == '__main__':
    bow = BOW('./data/train.tsv', mode='r')
    train_set, test_set = train_test_split(bow)
    print(len(train_set))
    for X, y in dataloader_bow(bow, train_set, 32):
        print(X.shape, y.shape)
        break
