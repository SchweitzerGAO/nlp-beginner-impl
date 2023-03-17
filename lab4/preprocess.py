import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data.dataset as Dataset
from torch.utils.data.dataloader import DataLoader
import pickle as pkl
from public.misc import Vocab, truncate_pad

'''
extract useful data from the original dataset
'''


def extract_data(path_r, path_w):
    data = []
    with open(path_r, 'r') as f:
        keys = []
        values = []
        for i, line in enumerate(f):
            if i >= 2:
                if line != '\n':
                    kk = line.split()[0].lower()
                    keys.append(kk)
                    vv = line.split()[3]
                    values.append(vv)
                else:
                    data.append([keys, values])
                    keys = []
                    values = []
    for sent_dict in data:
        values = sent_dict[1]
        for i in range(len(values)):
            now = values[i]
            if now[0] == 'I':
                length = 1
                idx = i
                while i + 1 < len(values) and values[i + 1] == now:
                    length += 1
                    i += 1
                if length == 1:
                    values[idx] = 'S-' + now.split('-')[1]
                else:
                    values[idx] = 'B-' + now.split('-')[1]
                    for j in range(idx + 1, idx + length - 1):
                        values[j] = 'M-' + now.split('-')[1]
                    values[idx + length - 1] = 'E-' + now.split('-')[1]

    with open(path_w, 'w') as f:
        for sent in data:
            f.write('<bos> O\n')
            for i in range(len(sent[0])):
                if 'docstart' not in sent[0][i]:
                    f.write(sent[0][i] + ' ' + sent[1][i] + '\n')
            f.write('<eos> O\n\n')


def read_data(path):
    sentences = []
    labels = []
    with open(path, 'r') as f:
        sent = []
        label = []
        for line in f:
            if line != '\n':
                sent.append(line.split()[0])
                label.append(line.split()[1])
            else:
                sentences.append(sent)
                labels.append(label)
                sent = []
                label = []
    return sentences, labels


'''
make the data digitalized 
'''


def collate_fn_lstm(data):
    chars, sentences, labels = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
    sentences.sort(key=lambda x: len(x), reverse=True)
    labels.sort(key=lambda x: len(x), reverse=True)
    chars.sort(key=lambda x: len(x), reverse=True)

    # lengths_sentence = torch.tensor([s.size(0) for s in sentences])

    chars = pad_sequence(chars, batch_first=True, padding_value=1.0)

    sentences = pad_sequence(sentences, batch_first=True, padding_value=1)
    # sentences = pack_padded_sequence(sentences, lengths_sentence, batch_first=True)

    labels = pad_sequence(labels, batch_first=True)

    return chars, sentences, labels


def collate_fn_cnn(chars, sentences, labels, max_sent, max_chars):
    length = len(chars)
    for i in range(length):
        if chars[i].size(0) < max_sent:
            size = chars[i].size(0)
            chars[i] = torch.cat((chars[i], torch.ones((max_sent - size, max_chars))), dim=0)
            sentences[i] = torch.cat((sentences[i], torch.ones((max_sent - size))), dim=0)
            labels[i] = torch.cat((labels[i], torch.zeros(max_sent - size)), dim=0)

    chars = [c.tolist() for c in chars]
    chars = torch.tensor(chars)

    sentences = [s.tolist() for s in sentences]
    sentences = torch.tensor(sentences)

    labels = [l.tolist() for l in labels]
    labels = torch.tensor(labels)

    return chars, sentences, labels


def flatten_sentence(tokens):
    return [word for line in tokens for word in line]


def tokenize_char(sentences):
    ret = []
    for line in sentences:
        temp = []
        for word in line:
            temp.append(list(word))
        ret.append(temp)

    return ret


def flatten_char(tokens):
    ret = []
    for sent in tokens:
        for words in sent:
            for word in words:
                ret += word
    return ret


class CONLLDataset(Dataset.Dataset):
    def __init__(self, sentences, labels, char_vocab=None,
                 sentence_vocab=None, label_vocab=None, char_embed='lstm'):
        chars = tokenize_char(sentences)
        tokens_sentence = flatten_sentence(sentences)
        tokens_char = flatten_char(chars)
        tokens_label = flatten_sentence(labels)

        self.char_vocab = Vocab(tokens_char, min_freq=5,
                                reserved_tokens=['<pad>']) if char_vocab is None else char_vocab
        self.sentence_vocab = Vocab(tokens_sentence, min_freq=5,
                                    reserved_tokens=['<pad>']) if sentence_vocab is None else sentence_vocab
        self.label_vocab = Vocab(tokens_label, has_unk=False) if label_vocab is None else label_vocab

        self.chars = []
        size_chars = []
        for sent in chars:
            temp_token = []
            temp_len = []
            for word in sent:
                token = torch.FloatTensor(self.char_vocab[word])
                temp_token.append(token)
                temp_len.append(token.size(0))
            self.chars.append(temp_token)
            size_chars.append(max(temp_len))
        self.max_chars = max(size_chars)
        padded_chars = []
        for sent in self.chars:
            temp = []
            for word in sent:
                word = truncate_pad(list(word), self.max_chars, self.char_vocab['<pad>'])
                temp.append(word)
            temp = [t.tolist() for t in temp]
            padded_chars.append(torch.tensor(temp))

        self.chars = padded_chars
        self.sentences = [torch.LongTensor(self.sentence_vocab[token]) for token in sentences]
        self.max_sent = max([s.size(0) for s in self.sentences])
        self.labels = [torch.LongTensor(self.label_vocab[token]) for token in labels]
        if char_embed == 'cnn':
            self.chars, self.sentences, self.labels = collate_fn_cnn(self.chars, self.sentences, self.labels, self.max_sent,
                                                                     self.max_chars)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.chars[idx], self.sentences[idx], self.labels[idx]


def load_train_data(batch_size=32, num_workers=0, char_embed='lstm'):
    sentences, labels = read_data('./data/train.txt')
    with open('./char_vocab.pkl', 'rb') as f:
        char_vocab = pkl.load(f)
    with open('./sentence_vocab.pkl', 'rb') as f:
        sentence_vocab = pkl.load(f)
    with open('./label_vocab.pkl', 'rb') as f:
        label_vocab = pkl.load(f)
    train_set = CONLLDataset(sentences, labels, char_vocab, sentence_vocab, label_vocab, char_embed)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_lstm if char_embed == 'lstm' else None,
                              num_workers=num_workers, drop_last=True)
    return train_loader, (char_vocab, sentence_vocab, label_vocab), train_set.max_sent, train_set.max_chars


if __name__ == '__main__':
    extract_data('./data/eng.testb', './data/testb.txt')
    # train_loader, vocabs, max_sent, max_chars = load_train_data(char_embed='cnn')
    #
    # for C, S, y in train_loader:
    #     pass
