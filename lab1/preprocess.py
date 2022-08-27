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
        raise NotImplementedError

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
            for phrase in self.corpus:
                phrase = phrase.lower()
                self.vocab = list(set(self.vocab + phrase.split(' ')))
            self.vocab.remove('')
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
            for sentence in list(self.corpus):
                sentence = sentence.lower()
                words = sentence.split()
                for word in words:
                    cnt = word_count.get(word, None)
                    if cnt is None:
                        word_count[word] = words.count(word)
                    else:
                        word_count[word] += words.count(word)
            word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            self.vocab = [tpl[0] for tpl in word_count[:max_features]]
            if '' in self.vocab:
                self.vocab.remove('')
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

    def voc2idx(self, voc):
        return self.idx.get(voc, -1)

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
        pass

    def choose_feature(self, max_features, mode, data_path):
        pass

    def voc2idx(self, voc):
        pass

    def generate_feature(self, phrase):
        pass


def train_test_split(feature_extractor, test_ratio=0.2):
    len_all = len(feature_extractor.corpus)
    len_train = int(len_all * (1 - test_ratio))
    train_set = feature_extractor.corpus[:len_train]
    train_label = feature_extractor.cls[:len_train]
    test_set = feature_extractor.corpus[len_train + 1:]
    test_label = feature_extractor.cls[len_train + 1:]
    return list(train_set), list(test_set), list(train_label), list(test_label)


def dataloader(feature_extractor, data, labels, batch_size, shuffle=True):
    len_train = len(data)
    num_batch = len_train // batch_size
    for i in range(0, num_batch * batch_size, batch_size):
        X = np.array(
            [feature_extractor.generate_feature(phrase) for phrase in data[i:i + batch_size]], dtype=np.float64)
        label = np.array(labels[i:i + batch_size])
        if shuffle:
            concat = np.concatenate((X, label.reshape(batch_size, -1)), axis=1)
            np.random.shuffle(concat)
            X = concat[:, 0:-1]
            label = concat[:, -1]
        y = list(map(lambda x: int(x - 1), label))
        y = np.eye(feature_extractor.num_cls, dtype=np.float64)[y]
        yield X, y


'''
test code
'''
if __name__ == '__main__':
    bow = BOW(max_features=1500, data_path='./proceeded_data/bow_1500.pkl', mode='w')
    train_set, test_set, train_label, test_label = train_test_split(bow)
    print(len(train_set))
    for X, y in dataloader(bow, train_set, train_label, 32):
        print(X.reshape((32, 1, -1)).shape, y[0].reshape(1, -1).shape)
        break
