import torch
import torch.nn as nn

from preprocess import load_train_data
from public.misc import PretrainedEmbedding


class BiLSTMEmbed(nn.Module):
    def __init__(self, word_length, sent_vocab, embed_size=100):
        super().__init__()
        self.bi_lstm = nn.LSTM(word_length, word_length // 2, batch_first=True, bidirectional=True)
        self.pretrained_embed = nn.Embedding(len(sent_vocab), embed_size)
        # wv = PretrainedEmbedding('../public/glove_6B_100d.pkl')
        # weight_embed = wv[sent_vocab.idx_to_token]
        # self.pretrained_embed.weight.data.copy_(weight_embed)
        # self.pretrained_embed.weight.requires_grad = False

    def forward(self, C, S):
        output, _ = self.bi_lstm(C)
        pretrained = self.pretrained_embed(S)
        embedded = torch.cat((output, pretrained), dim=2)
        return embedded


class CNNEmbed(nn.Module):
    def __init__(self, char_vocab, sent_vocab, word_length, window_size=3, filters=30, embed_size=100):
        super().__init__()
        self.char_embed = nn.Embedding(len(char_vocab), word_length)
        self.dropout = nn.Dropout(0.5)
        # self.cnn = nn.Conv1d(,filters,kernel_size=window_size)
        self.max_pool = nn.MaxPool1d(word_length)
        self.pretrained_embed = nn.Embedding(len(sent_vocab), embed_size)

    def forward(self, C, S):
        pass


class Encoder(nn.Module):
    pass


if __name__ == '__main__':
    train_loader, vocabs = load_train_data()
    word_length = 61
    hidden_size_lstm_embed = 25
    sent_vocab = vocabs[1]
    lstm_embed = BiLSTMEmbed(word_length, sent_vocab)
    for (C, S), y in train_loader:
        vector = lstm_embed(C, S)
        pass
