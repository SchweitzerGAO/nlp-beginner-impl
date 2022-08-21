import numpy as np
from preprocess import BOW, dataloader_bow


class ScratchTextClassifier:
    def __init__(self, bow):
        self.len_vocab = len(bow.vocab)
        self.len_label = len(set(bow.label))
