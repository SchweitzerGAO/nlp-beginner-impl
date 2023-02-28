import collections
import torch

# count the number that each character appears
def count_corpus(tokens):
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def truncate_pad(sentence_idx, length, pad_idx):
    while len(sentence_idx) < length:
        sentence_idx.append(pad_idx)
    return torch.Tensor(sentence_idx[:length])


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        self.unk = 0
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        # descending by number of appearance
        self.token_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
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
