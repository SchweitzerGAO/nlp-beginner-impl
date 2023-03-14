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
        self.embed_size = embed_size + (word_length // 2) * 2

    def forward(self, C, S):
        output, _ = self.bi_lstm(C)
        pretrained = self.pretrained_embed(S)
        embedded = torch.cat((pretrained, output), dim=2)
        return embedded


class CNNEmbed(nn.Module):
    def __init__(self, char_vocab, sent_vocab, word_length, sent_length, window_size=3, embed_size=100):
        super().__init__()
        self.char_embed = nn.Embedding(len(char_vocab), word_length)
        self.dropout = nn.Dropout(0.5)
        self.cnn = nn.Conv2d(sent_length, sent_length, kernel_size=window_size)
        out_length = word_length - window_size + 1
        self.max_pool = nn.MaxPool1d(out_length)
        self.pretrained_embed = nn.Embedding(len(sent_vocab), embed_size)
        self.embed_size = embed_size + out_length

    def forward(self, C, S):
        output = self.char_embed(C.long())
        output = self.cnn(self.dropout(output))
        pool_list = []
        for i in range(output.size(2)):
            pool_list.append(self.max_pool(output[:, :, i, :]))
        output = torch.stack(pool_list, dim=2).squeeze(3)
        pretrained = self.pretrained_embed(S.long())
        embedded = torch.cat((pretrained, output), dim=2)
        return embedded


class Encoder(nn.Module):
    pass


if __name__ == '__main__':
    train_loader, vocabs, max_sent, max_chars = load_train_data(char_embed='cnn')
    lstm_embed = BiLSTMEmbed(max_chars, vocabs[1])
    cnn_embed = CNNEmbed(vocabs[0], vocabs[1], max_chars, max_sent)
    for C, S, y in train_loader:
        vector = cnn_embed(C, S)
        pass
