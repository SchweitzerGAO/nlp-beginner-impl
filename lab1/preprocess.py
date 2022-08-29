import pandas as pd
import numpy as np
import pickle as pkl

'''
Base class
'''


class FeatureExtractor:
    def __init__(self, train_path='./data/train.tsv', max_features=None):
        df = pd.read_csv(train_path, sep='\t')
        self.corpus = df['Phrase']
        self.cls = df['Sentiment']
        self.max_features = max_features
        self.vocab = []  # index-vocabulary
        self.num_cls = len(set(self.cls))
        self.idx = dict()  # vocabulary-index

    def generate_data(self, mode, data_path):
        raise NotImplementedError

    def choose_feature(self, max_features, mode, data_path):
        raise NotImplementedError

    def voc2idx(self, voc):
        return self.idx.get(voc, -1)

    def generate_feature(self, phrase):
        raise NotImplementedError


'''
Bag of Words
'''


class BOW(FeatureExtractor):
    def __init__(self, max_features=None, data_path='./proceeded_data/bow.pkl', mode='r'):
        super().__init__(max_features=max_features)
        if self.max_features is None:
            self.generate_data(mode, data_path)
        else:
            self.choose_feature(self.max_features, mode, data_path)

    def generate_data(self, mode, data_path):
        if mode == 'w':
            vocab = set()
            for phrase in list(self.corpus):
                phrase = phrase.lower()
                words = phrase.split()
                vocab.update(word for word in words if word.isalpha())
            self.vocab = list(vocab)
            for i, voc in enumerate(self.vocab):
                self.idx[voc] = i
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

    def choose_feature(self, max_features, mode, data_path):
        if mode == 'w':
            word_count = dict()
            for phrase in list(self.corpus):
                phrase = phrase.lower()
                words = phrase.split(' ')
                for word in words:
                    if word.isalpha():
                        cnt = word_count.get(word, None)
                        if cnt is None:
                            word_count[word] = words.count(word)
                        else:
                            word_count[word] += words.count(word)
            word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            self.vocab = [tpl[0] for tpl in word_count[51:51 + max_features]]  # remove the stop words
            for i, voc in enumerate(self.vocab):
                self.idx[voc] = i
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

    def generate_feature(self, phrase):
        bag = np.array([0.] * len(self.vocab), dtype=np.float32)
        words = phrase.split()
        for word in words:
            word_idx = self.voc2idx(word)
            if word_idx != -1 and bag[word_idx] == 0:
                bag[word_idx] = float(words.count(word))
        return bag


'''
N-gram
'''


class NGram(FeatureExtractor):
    def __init__(self, n, max_features=None, data_path='./proceeded_data/ngram.pkl', mode='r'):
        super().__init__(max_features=max_features)
        self.n = n
        if self.max_features is None:
            self.generate_data(mode, data_path)
        else:
            self.choose_feature(self.max_features, mode, data_path)

    def generate_data(self, mode, data_path):
        if mode == 'w':
            vocab = set()
            for phrase in list(self.corpus):
                phrase = phrase.lower()
                words = phrase.split(' ')
                slices = [' '.join(words[i:i + self.n])
                          for i in range(len(words) + 1 - self.n)
                          if all(s.isalpha()
                                 for s in words[i:i + self.n])]
                vocab.update(slices)
            self.vocab = list(vocab)
            for i, voc in enumerate(self.vocab):
                self.idx[voc] = i
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

    def choose_feature(self, max_features, mode, data_path):
        if mode == 'w':
            sli_count = dict()
            for phrase in list(self.corpus):
                phrase = phrase.lower()
                words = phrase.split(' ')
                slices = [' '.join(words[i:i + self.n])
                          for i in range(len(words) + 1 - self.n)
                          if all(s.isalpha()
                                 for s in words[i:i + self.n])]
                for sli in slices:
                    cnt = sli_count.get(sli, None)
                    if cnt is None:
                        sli_count[sli] = slices.count(sli)
                    else:
                        sli_count[sli] += slices.count(sli)
            sli_count = sorted(sli_count.items(), key=lambda x: x[1], reverse=True)
            self.vocab = [tpl[0] for tpl in sli_count[51:max_features + 51]]
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

    def generate_feature(self, phrase):
        bag = np.array([0.] * len(self.vocab), dtype=np.float32)
        words = phrase.split(' ')
        slices = [' '.join(words[i:i + self.n])
                  for i in range(len(words) + 1 - self.n)]
        for sli in slices:
            slice_idx = self.voc2idx(sli)
            if slice_idx != -1 and bag[slice_idx] == 0:
                bag[slice_idx] = float(slices.count(sli))
        return bag


def train_test_split(feature_extractor, test_ratio=0.2, shuffle=True):
    len_all = len(feature_extractor.corpus)
    len_train = int(len_all * (1 - test_ratio))
    dataset = feature_extractor.corpus
    gt = feature_extractor.cls
    if shuffle:
        dataset = np.array(list(dataset))
        gt = [int(label) for label in list(gt)]
        gt = np.array(gt)
        concat = np.concatenate((dataset.reshape(-1, 1), gt.reshape(-1, 1)), axis=1)
        np.random.shuffle(concat)
        dataset = concat[:, 0]
        gt = concat[:, 1]

    train_set = dataset[:len_train]
    train_label = gt[:len_train]
    test_set = dataset[len_train + 1:]
    test_label = gt[len_train + 1:]
    return list(train_set), list(test_set), list(train_label), list(test_label)


def dataloader(feature_extractor, data, labels, batch_size):
    len_train = len(data)
    num_batch = len_train // batch_size
    for i in range(0, num_batch * batch_size, batch_size):
        X = np.array(
            [feature_extractor.generate_feature(phrase) for phrase in data[i:i + batch_size]], dtype=np.float64)
        label = np.array(labels[i:i + batch_size])
        y = list(map(lambda x: int(x) - 1, label))
        y = np.eye(feature_extractor.num_cls, dtype=np.float64)[y]
        yield X, y


'''
test code
'''
if __name__ == '__main__':
    bow_5000 = BOW(data_path='./proceeded_data/bow_5000.pkl')
    train_set, test_set, train_label, test_label = train_test_split(bow_5000)
    print(len(train_set))
    for X, y in dataloader(bow_5000, train_set, train_label, 32):
        print(X.reshape((32, 1, -1)).shape, y[0].reshape(1, -1).shape)
        break
