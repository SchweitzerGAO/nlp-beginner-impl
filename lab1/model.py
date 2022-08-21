import numpy as np
from preprocess import BOW, dataloader_bow


class ScratchTextClassifier:
    def __init__(self, len_vocab, num_cls, loss='ce', num_hidden=256):
        self.len_vocab = len_vocab
        self.num_cls = num_cls
        self.num_hidden = num_hidden
        self.weights = [np.random.randn(self.len_vocab, self.num_hidden),
                        np.random.randn(self.num_hidden, self.num_cls)]
        self.biases = [np.zeros(1, self.num_hidden), np.zeros(1, self.num_cls)]
        self.output = []
        self.activated_output = []
        self.loss = loss

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return None
