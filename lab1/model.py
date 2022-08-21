import numpy as np
from preprocess import BOW, dataloader_bow



class ScratchTextClassifier:
    def __init__(self, bow,num_hidden=256):
        self.len_vocab = len(bow.vocab)
        self.num_cls = bow.num_cls




    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return None
