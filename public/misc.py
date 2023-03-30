import collections
import torch
import pickle as pkl


# count the number that each character appears
def count_corpus(tokens):
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# pad or truncate the input sentence(s)
def truncate_pad(sentence_idx, length, pad_idx):
    while len(sentence_idx) < length:
        sentence_idx.append(pad_idx)
    return torch.tensor(sentence_idx[:length])


# gradient clipping
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None, has_unk=True):
        self.unk = 0 if has_unk else None
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        # descending by number of appearance
        self.token_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens if has_unk else reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_sorted:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)  # a list of descending order by appearance
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # an inverse operation of idx_to_token

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


class PretrainedEmbedding:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.word_vectors = pkl.load(f)
        self.word_vectors['<unk>'] = torch.zeros_like(self.word_vectors['the'])
        self.word_vectors['<pad>'] = torch.zeros_like(self.word_vectors['the'])

    def __getitem__(self, words):
        return torch.stack([self.word_vectors.get(word, self.word_vectors['<unk>']) for word in words])

    def __len__(self):
        return len(self.word_vectors)


if __name__ == '__main__':
    wv = PretrainedEmbedding('./glove_6B_100d.pkl')
    pass
